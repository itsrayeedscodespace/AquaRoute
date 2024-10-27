from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pyarrow as pa
import pyarrow.feather as feather

def create_ml_enhanced_interpolation_model(spark):
    # Read the CSV data
    df = spark.read.csv("/FileStore/tables/Level012R1.csv", header=True)
    
    # Convert relevant columns to numeric
    numeric_cols = [
        "longitude (degree: E+, W-)", 
        "latitude (degree: N+, S-)",
        "Level 0 (pieces/m3)",
        "Level 1 (pieces/m3)",
        "Level 2p (pieces/km2)",
        "Level 2w1 (g/km2)",
        "Level 2w2 (g/km2)",
        "windspeed (m/s)",
        "significant wave height (m)"
    ]
    
    for col in numeric_cols:
        df = df.withColumn(col, col.cast("double"))
    
    # Create a standardized grid
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -90, 90
    grid_spacing = 1.0
    
    lon_grid = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)
    lat_grid = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    def ml_enhanced_interpolation(param_name):
        # Convert to pandas for processing
        pdf = df.select(
            "longitude (degree: E+, W-)",
            "latitude (degree: N+, S-)",
            "windspeed (m/s)",
            "significant wave height (m)",
            param_name
        ).toPandas()
        
        # Remove rows with null values
        pdf = pdf.dropna()
        
        if len(pdf) < 10:  # Need minimum samples for ML
            return None
            
        # Prepare features for ML
        X = pdf[["longitude (degree: E+, W-)", "latitude (degree: N+, S-)", 
                "windspeed (m/s)", "significant wave height (m)"]].values
        y = pdf[param_name].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf_model.fit(X_scaled, y)
        
        # Create RBF interpolator for residuals
        rbf = Rbf(
            pdf["longitude (degree: E+, W-)"],
            pdf["latitude (degree: N+, S-)"],
            y - rf_model.predict(X_scaled),
            function='multiquadric'
        )
        
        # Prepare grid points for prediction
        grid_points = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))
        # Add dummy values for windspeed and wave height (median values)
        grid_points = np.column_stack((
            grid_points,
            np.full(grid_points.shape[0], pdf["windspeed (m/s)"].median()),
            np.full(grid_points.shape[0], pdf["significant wave height (m)"].median())
        ))
        
        # Scale grid points
        grid_points_scaled = scaler.transform(grid_points)
        
        # Combine ML prediction and RBF interpolation of residuals
        ml_pred = rf_model.predict(grid_points_scaled)
        residual_pred = rbf(grid_points[:, 0], grid_points[:, 1])
        
        combined_pred = (ml_pred + residual_pred).reshape(lon_mesh.shape)
        
        # Ensure non-negative values for concentrations
        combined_pred = np.maximum(combined_pred, 0)
        
        return combined_pred
    
    # Create interpolated grids for each parameter
    interpolated_data = {}
    for param in ["Level 0 (pieces/m3)", "Level 1 (pieces/m3)", "Level 2p (pieces/km2)"]:
        interpolated_data[param] = ml_enhanced_interpolation(param)
    
    # Convert to DataFrame
    rows = []
    for i in range(len(lat_grid)):
        for j in range(len(lon_grid)):
            row = {
                'longitude': lon_grid[j],
                'latitude': lat_grid[i]
            }
            for param, grid in interpolated_data.items():
                if grid is not None:
                    row[param.replace(" ", "_")] = float(grid[i, j])
            rows.append(row)
    
    # Create pandas DataFrame
    pdf_final = pd.DataFrame(rows)
    
    # Save as feather file
    feather.write_feather(pdf_final, '/dbfs/FileStore/crush.feather')
    
    # Create Spark DataFrame for additional processing if needed
    schema = StructType([
        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True)
    ])
    for param in interpolated_data.keys():
        schema.add(param.replace(" ", "_"), DoubleType(), True)
    
    return spark.createDataFrame(rows, schema)

def main():
    spark = SparkSession.builder \
        .appName("ML Enhanced Marine Data Interpolation") \
        .getOrCreate()
    
    # Create interpolated data
    interpolated_df = create_ml_enhanced_interpolation_model(spark)
    
    # Example query to verify results
    interpolated_df.select("longitude", "latitude", "Level_0_(pieces/m3)") \
        .show(5)

if __name__ == "__main__":
    main()
