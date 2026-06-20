import logging
import os
import pandas as pd
import numpy as np

from steps.ingest_data import ingest_data
from steps.split_data import split_data
from steps.clean_data import clean_data
from steps.feature_engineering import feature_engineer
from steps.handle_missing import handle_missing
from steps.handle_outliers import handle_outliers
from steps.encode_categorical import encode_categorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(raw_path: str = "data/raw/used_cars.csv", output_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the end-to-end data preprocessing pipeline and returns the train and test DataFrames."""
    logger.info("Starting end-to-end training pipeline...")
    
    # Ingest
    df = ingest_data(raw_path)
    logger.info("Ingested data shape: %s", df.shape)

    df = df.drop(index = 693)
    # df = df.drop(index = [  18,   69,   85,   93,  114,  158,  175,  233,  251,  276,  289,
    #     320,  333,  348,  351,  394,  406,  434,  464,  465,  485,  497,
    #     498,  500,  502,  528,  559,  563,  583,  602,  634,  693,  697,
    #     704,  721,  731,  737,  739,  747,  781,  783,  807,  813,  853,
    #     869,  889,  900,  922,  924,  930,  967,  974,  983,  988,  992,
    #    1003, 1035, 1057, 1067, 1086, 1099, 1134, 1139, 1161, 1166, 1181,
    #    1182, 1202, 1203, 1206, 1240, 1248, 1270, 1298, 1302, 1318, 1321,
    #    1322, 1328, 1351, 1361, 1380, 1386, 1458, 1477, 1497, 1513, 1543,
    #    1545, 1548, 1557, 1601, 1612, 1641, 1658, 1707, 1710, 1724, 1727,
    #    1729, 1736, 1752, 1767, 1792, 1805, 1818, 1819, 1852, 1866, 1881,
    #    1907, 1916, 1939, 1954, 1984, 2020, 2025, 2029, 2033, 2048, 2061,
    #    2068, 2101, 2106, 2107, 2112, 2118, 2119, 2157, 2174, 2241, 2253,
    #    2264, 2273, 2277, 2284, 2297, 2350, 2359, 2361, 2409, 2417, 2435,
    #    2497, 2521, 2537, 2549, 2559, 2562, 2567, 2593, 2614, 2638, 2645,
    #    2664, 2670, 2681, 2751, 2756, 2766, 2769, 2775, 2791, 2826, 2827,
    #    2831, 2840, 2857, 2888, 2901, 2997, 2998, 3002, 3028, 3038, 3043,
    #    3056, 3059, 3066, 3080, 3087, 3117, 3141, 3158, 3164, 3169, 3184,
    #    3202])
    
    # Split
    data = split_data(df)
    logger.info("Split data into train: %s and test: %s", data[0].shape, data[1].shape)

    # removing influential rows in the train df
    train_df, test_df = data
    
    train_df = train_df.drop(index =[ 559,  251, 1240,  434,  583, 2118,  320, 2645, 3038, 1181, 2614, 1866,
       2361, 2119, 1805, 1707, 1658, 2068, 2157, 1545, 1270,  737, 2020,  233,
       2537, 1321, 1852, 2025, 2521, 2827,  464,  697,  351, 2112,  502,  704,
       1727, 2857, 2241, 2766, 2497,  739,  634, 1752,  721, 2681, 1380,  930,
        900, 2297,  528, 1818, 3164, 1322, 1318, 2350, 2664,   93,   18, 1819,
       2998, 1601,  276,   85, 1298, 2264, 1612, 2559, 1767, 2593, 1477, 1206,
       2567, 3169, 3158,  783,  988, 2409, 3080,  813, 1035, 1003, 1202, 1182,
         69, 2751, 3117, 1548,  158,  924, 1916, 2791, 2549, 2769,  175, 3066,
       2756, 2174, 2901, 3141],errors='ignore')
    
    data = train_df, test_df
    # Clean
    data = clean_data(data)
    logger.info("Cleaned data shape - train: %s, test: %s", data[0].shape, data[1].shape)
    
    # Feature Engineer
    data = feature_engineer(data)
    logger.info("Feature engineered shape - train: %s, test: %s", data[0].shape, data[1].shape)
    
    # Handle Missing Values
    data = handle_missing(data)
    logger.info("Imputed missing values shape - train: %s, test: %s", data[0].shape, data[1].shape)
    
    # Handle Outliers (IQR)
    data = handle_outliers(data)
    logger.info("Outliers handled (IQR) shape - train: %s, test: %s", data[0].shape, data[1].shape)
    

    # Categorical Encoding
    data = encode_categorical(data)
    logger.info("Categorically encoded shape - train: %s, test: %s", data[0].shape, data[1].shape)


    # scale the data
    pipeline = Pipeline([   
    ('scaler', MinMaxScaler())
    ])

    train_df, test_df = data

    # 2. Force the pipeline to output Pandas DataFrames
    pipeline.set_output(transform="pandas")

    # 3. Fit and transform your data
    train_df = pipeline.fit_transform(train_df)
    test_df = pipeline.transform(test_df)
    
    # Create a column for mileage, 
    # threshold = 0.75
    # train_df['is_high_mileage'] = (train_df['mileage_num'] >= threshold).astype(float)
    # test_df['is_high_mileage'] = (test_df['mileage_num'] >= threshold).astype(float)

    # # Another switch
    # df['mileage_above_threshold'] = np.maximum(0,train_df['mileage_num'] - threshold)
    # df['mileage_above_threshold'] = np.maximum(0, test_df['mileage_num'] - threshold)

    # Save the files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=True)
    test_df.to_csv(test_path, index=True)
    
    logger.info("Train dataset successfully saved to %s", train_path)
    logger.info("Test dataset successfully saved to %s", test_path)
    
    logger.info(f"Train_df: {train_df.shape}, Test_df :{test_df.shape}")
    return train_df, test_df


if __name__ == "__main__":
    run_training_pipeline()
