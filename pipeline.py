from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# PREPROCESSING
class Preprocessing:

    idCols = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN']
    numericPlaceholderCols = ['Age', 'Annual_Income', 'Outstanding_Debt',
                                 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Monthly_Balance']
    ordinalCol = ['Credit_Mix']
    ordinalOrder = ['Bad', 'Standard', 'Good']

    def __init__(self):
        self.numCol = None
        self.catCol = None
        self.ordCol = self.ordinalCol
        self.nomCol = None
        self.transformer = None

    @staticmethod
    def _parse_history(val):
        if pd.isna(val):
            return np.nan
        try:
            parts = str(val).split(' ')
            return int(parts[0]) * 12 + int(parts[3])
        except Exception:
            return np.nan

    def clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=[c for c in self.idCols if c in df.columns])

        for col in self.numericPlaceholderCols:
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip('_'), errors='coerce')

        df['Amount_invested_monthly'] = pd.to_numeric(
            df['Amount_invested_monthly'].astype(str).str.replace('_', '', regex=False), errors='coerce')
        df['Changed_Credit_Limit'] = pd.to_numeric(
            df['Changed_Credit_Limit'].replace('_', np.nan), errors='coerce')
        df['Monthly_Balance'] = pd.to_numeric(
            df['Monthly_Balance'].astype(str).str.replace('_', '', regex=False), errors='coerce')

        df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)

        df.loc[(df['Age'] < 14) | (df['Age'] > 100), 'Age'] = np.nan
        df.loc[df['Interest_Rate'] > 40, 'Interest_Rate'] = np.nan
        df.loc[(df['Num_of_Loan'] == -100) | (df['Num_of_Loan'] > 10), 'Num_of_Loan'] = np.nan
        df.loc[(df['Num_Bank_Accounts'] < 0) | (df['Num_Bank_Accounts'] > 20), 'Num_Bank_Accounts'] = np.nan
        df.loc[df['Num_Credit_Card'] > 15, 'Num_Credit_Card'] = np.nan
        df.loc[df['Num_of_Delayed_Payment'] > 30, 'Num_of_Delayed_Payment'] = np.nan
        df.loc[df['Num_Credit_Inquiries'] > 20, 'Num_Credit_Inquiries'] = np.nan
        df.loc[df['Total_EMI_per_month'] > 5000, 'Total_EMI_per_month'] = np.nan
        df.loc[df['Monthly_Balance'] < -10000, 'Monthly_Balance'] = np.nan

        df['Credit_History_Months'] = df['Credit_History_Age'].apply(self._parse_history)
        df = df.drop(columns=['Credit_History_Age'])

        df['Loan_Type_Count'] = df['Type_of_Loan'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        df = df.drop(columns=['Type_of_Loan'])

        return df

    def fit(self, X: pd.DataFrame):
        self.numCol = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
        self.catCol = [c for c in X.columns if c not in self.numCol]
        self.nomCol = [c for c in self.catCol if c not in self.ordCol]

        numTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        ordTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[self.ordinalOrder],
                                        handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        nomTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.transformer = ColumnTransformer(transformers=[
            ('num', numTransformer, self.numCol),
            ('ord', ordTransformer, self.ordCol),
            ('nom', nomTransformer, self.nomCol)
        ], remainder='passthrough')

        self.transformer.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        return self.transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    def get_feature_names(self):
        return self.transformer.get_feature_names_out()


# TRAINING
class Training:

    modelRegistry = {
        'logistic_regression': lambda p: LogisticRegression(max_iter=1000, class_weight='balanced', **p),
        'decision_tree':       lambda p: DecisionTreeClassifier(class_weight='balanced', **p),
        'random_forest':       lambda p: RandomForestClassifier(class_weight='balanced', **p),
        'svm':                 lambda p: SVC(kernel='rbf', class_weight='balanced', **p),
        'gradient_boosting':   lambda p: GradientBoostingClassifier(**p),
        'xgboost':             lambda p: XGBClassifier(eval_metric='mlogloss', **p),
    }

    def __init__(self, modelName: str, params: dict = None):
        if modelName not in self.modelRegistry:
            raise ValueError(f"Unknown modelName '{modelName}'. "
                              f"Choose one of {list(self.modelRegistry)}")
        self.modelName = modelName
        self.params = params or {}
        self.model = None

    def fit(self, xTrain, yTrain, sampleWeight=None):
        self.model = self.modelRegistry[self.modelName](self.params)
        if self.modelName in ('xgboost', 'gradient_boosting') and sampleWeight is not None:
            self.model.fit(xTrain, yTrain, sample_weight=sampleWeight)
        else:
            self.model.fit(xTrain, yTrain)
        return self.model


# EVALUATION
class Evaluation:

    def __init__(self, targetMapping: dict, positiveClass: str = 'Poor'):
        self.targetMapping = targetMapping
        self.positiveClass = positiveClass
        self.positiveIndex = targetMapping[positiveClass]

    def evaluate(self, model, xTest, yTestEncoded) -> dict:
        preds = model.predict(xTest)
        report = classification_report(
            yTestEncoded, preds,
            output_dict=True, zero_division=0
        )
        return {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            f'{self.positiveClass.lower()}_recall': report[str(self.positiveIndex)]['recall'],
            f'{self.positiveClass.lower()}_precision': report[str(self.positiveIndex)]['precision'],
            'report': report,
        }


# ORCHESTRATION
modelConfigs = {
    'logistic_regression': {},
    'decision_tree':       {'random_state': 42},
    'random_forest':       {'n_estimators': 300, 'max_depth': 25,
                             'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42},
    'svm':                 {'random_state': 42},
    'gradient_boosting':   {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
    'xgboost':             {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
                             'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42},
}


def run_pipeline(dataPath: str, experimentName: str = 'creditScoreAssessment',
                  outputModelPath: str = 'final_model_pipeline.pkl',
                  outputMappingPath: str = 'target_mapping.pkl'):

    print(f"[1/5] Loading raw data from {dataPath} ...")
    raw = pd.read_csv(dataPath)

    preprocessing = Preprocessing()
    cleaned = preprocessing.clean_raw(raw)

    X = cleaned.drop(columns='Credit_Score')
    y = cleaned['Credit_Score']

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    targetMapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
    yTrainEnc = yTrain.map(targetMapping)
    yTestEnc = yTest.map(targetMapping)

    print("[2/5] Fitting preprocessing on the training fold only to prevent leakage ...")
    xTrainP = preprocessing.fit_transform(xTrain)
    xTestP = preprocessing.transform(xTest)
    print(f"      -> train shape {xTrainP.shape}, test shape {xTestP.shape}")

    evaluator = Evaluation(targetMapping, positiveClass='Poor')

    print(f"[3/5] Logging experiment runs to MLflow (experiment='{experimentName}') ...")
    mlflowDbPath = Path(outputModelPath).parent.resolve() / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflowDbPath}")
    mlflow.set_experiment(experimentName)

    results = []
    fittedModels = {}
    
    sampleWeights = compute_sample_weight('balanced', yTrainEnc)

    for modelName, params in modelConfigs.items():
        with mlflow.start_run(run_name=modelName):
            trainer = Training(modelName, params)
            model = trainer.fit(xTrainP, yTrainEnc, sampleWeight=sampleWeights)
            metrics = evaluator.evaluate(model, xTestP, yTestEnc)

            mlflow.log_param('model_name', modelName)
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric('accuracy', metrics['accuracy'])
            mlflow.log_metric('macro_f1', metrics['macro_f1'])
            mlflow.log_metric('poor_recall', metrics['poor_recall'])

            fittedModels[modelName] = model
            results.append({
                'model': modelName,
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'poor_recall': metrics['poor_recall'],
            })
            print(f"      {modelName:20s} | acc={metrics['accuracy']:.4f} "
                  f"| macroF1={metrics['macro_f1']:.4f} | poorRecall={metrics['poor_recall']:.4f}")

    resultsDf = pd.DataFrame(results).sort_values('macro_f1', ascending=False).reset_index(drop=True)
    print("\n[4/5] Baseline ranking by macro F1:")
    print(resultsDf.to_string(index=False))

    shortlist = resultsDf.head(2).reset_index(drop=True)
    bestRow = shortlist.sort_values('poor_recall', ascending=False).iloc[0]
    bestName = bestRow['model']
    bestModel = fittedModels[bestName]

    print(f"\n      Shortlist (top-2 by macro F1): {list(shortlist['model'])}")
    print(f"      Best model (highest Poor recall within shortlist): {bestName}")

    print("[5/5] Exporting best model as a full production pipeline ...")
    fullPipeline = Pipeline(steps=[
        ('preprocessing', preprocessing.transformer),
        ('classifier', bestModel)
    ])
    joblib.dump(fullPipeline, outputModelPath)
    joblib.dump(targetMapping, outputMappingPath)
    print(f"      Exported: {outputModelPath}")
    print(f"      Exported: {outputMappingPath}")

    with mlflow.start_run(run_name=f'BEST_{bestName}'):
        mlflow.log_param('best_model', bestName)
        mlflow.log_metric('poor_recall', bestRow['poor_recall'])
        mlflow.log_metric('macro_f1', bestRow['macro_f1'])
        mlflow.sklearn.log_model(fullPipeline, artifact_path='best_full_pipeline')

    print(f"\n MLflow UI: jalankan perintah berikut lalu buka http://localhost:5000")
    print(f"      mlflow ui --backend-store-uri sqlite:///{mlflowDbPath}")

    return resultsDf, bestName, fullPipeline, targetMapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and track credit-score models locally with MLflow.')
    parser.add_argument('--data', type=str, default='data_D.csv', help='Path to the raw csv file')
    parser.add_argument('--experiment', type=str, default='creditScoreAssessment', help='MLflow experiment name')
    parser.add_argument('--model-out', type=str, default='final_model_pipeline.pkl')
    parser.add_argument('--mapping-out', type=str, default='target_mapping.pkl')
    args = parser.parse_args()

    run_pipeline(args.data, args.experiment, args.model_out, args.mapping_out)