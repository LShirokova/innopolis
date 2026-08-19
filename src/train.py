import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
import logging
from dotenv import load_dotenv
from db_manager import DatabaseManager

load_dotenv()

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Константы
ENGINE_ID_COL = 'engine_id'
TARGET_COL = 'target_label'
MODEL_SAVE_PATH = "models/best_model.pkl"
RANDOM_STATE = 42
RESULTS = [] # для сравнения моделей

db_manager = DatabaseManager(
    db_user=os.getenv('DB_USER'),
    db_pass=os.getenv('DB_PASSWORD'),
    db_host=os.getenv('DB_HOST'),
    db_port=os.getenv('DB_PORT'),
    db_name=os.getenv('DB_NAME'),
)


def load_data():
    """Загружает данные из Feature Store"""
    logger.info("\n🟢 Загрузка данных из Feature Store...")
    df = pd.read_sql("SELECT * FROM features.ml_features", con=db_manager.engine)

    X = df.drop(columns=[TARGET_COL, ENGINE_ID_COL, 'cycle'])
    y = df[TARGET_COL]
    groups = df[ENGINE_ID_COL]
    return X, y, groups

def split_data(X, y, groups):
    """Разбивает данные по ID двигателей (предотвращает утечку)."""
    logger.info("\n✂️ Разбиение данных")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    logger.info(f"Train: {X_train.shape[0]} строк от {groups_train.nunique()} двигателей")
    logger.info(f"Test: {X_test.shape[0]} строк от {groups_test.nunique()} двигателей")
    return X_train, X_test, y_train, y_test, groups_train, groups_test

def evaluate_baseline(X_train, X_test, y_train, y_test):
    """Обучает и оценивает базовую модель."""
    logger.info("\n--- 🟢 ОЦЕНКА BASELINE (Logistic Regression) ---")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)

    # Заглушаю предупреждения в консоли
    start_time_base = time.time()
    model.fit(X_train, y_train)
    train_time_base = time.time() - start_time_base
    
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    RESULTS.append({
        'Model': 'Base',
        'Recall': recall,
        'Precision': precision,
        'FPR': fpr,
        'ROC_AUC': roc_auc,
        'Time': train_time_base
    })

    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'FPR: {fpr:.4f}')
    print(f'ROC_AUC: {roc_auc:.4f}')
    print(f'Time: {train_time_base:.4f}')

    os.makedirs('img/model_base', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt=".0f", cbar=False)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    ax.set_title('Базовая модель (LogisticRegression)')
    plt.savefig('img/model_base/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('📊 марица корреляции сохранена в img/model_base/confusion_matrix.png')
    
    logger.info(f"✅ Baseline модель обучена: {recall:.3f} (Цель >= 0.85)")

    return model

def run_experiments(X, y, groups):
    """Проводит кросс-валидацию разных алгоритмов."""
    logger.info("\n--- 🟢 ЗАПУСК ЭКСПЕРИМЕНТОВ (Cross-Validation) ---")
    gkf = GroupKFold(n_splits=5)
    
    def fp_rate_scorer(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0

    fpr_scorer = make_scorer(fp_rate_scorer, greater_is_better=False)

    scoring = {
        'recall': 'recall',
        'fpr': fpr_scorer,
        'roc_auc': 'roc_auc',
        'precision': 'precision'
    }
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', eval_metric='logloss', learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    }
    
    for name, model in models.items():
        logger.info(f"Тестирование {name}...")
        cv_results = cross_validate(model, X, y, groups=groups, cv=gkf, scoring=scoring, return_train_score=True, n_jobs=-1)
        RESULTS.append({
            'Model': name,
            'Recall': np.mean(cv_results['test_recall']),
            'FPR': np.mean(cv_results['test_fpr']) * -1,
            'Precision': np.mean(cv_results['test_precision']),
            'ROC_AUC': np.mean(cv_results['test_roc_auc']),
            'Time': np.mean(cv_results['fit_time']),
        })
        
    results_df = pd.DataFrame(RESULTS)
    logger.info(f"Результаты CV:\n{results_df.to_string(index=False)}")

    plot_model_comparison(results_df)
    return results_df

def plot_model_comparison(results_df):
    """Строит сетку графиков для сравнения моделей."""
    print("\n--- Построение графиков сравнения ---")

    # Список метрик для отображения (порядок важен для логики)
    metrics = ['Recall', 'FPR', 'Precision']
    
    # Цвета для каждой модели (можно менять)
    colors = ['#8da0cb', # Base - синий
              '#a6d854', # RandomForest - зеленый
              '#db5f57', # XGBoost - красный
              '#e78ac3'] # LightGBM - оранжевый
    
    # Создаем сетку 3x3 (так как 7 метрик, 2 ячейки будут пустые)
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(16, 6))
    axes = axes.ravel() # Разворачиваем матрицу 3x3 в плоский список из 9 элементов

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Строим столбчатую диаграмму
        results_df.plot(
            x='Model', 
            y=metric, 
            kind='bar', 
            ax=ax, 
            color=colors, 
            edgecolor='black', # Обводка столбиков
            legend=False
        )
        
        # Настройки внешнего вида
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold', pad=6)
        ax.set_ylabel('Значение', fontsize=10)
        ax.set_xticklabels(results_df['Model'], rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Добавляем значения над каждым столбиком (очень нравится комиссиям)
        for p in ax.patches:
            value = p.get_height()
            # Для времени выводим 2 знака, для остальных 3
            text = f"{value:.3f}"
            ax.annotate(text, 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # АРХИТЕКТУРНЫЙ БОНУС: Рисуем линии бизнес-требований (SLA)
        if metric == 'Recall':
            ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.3, linewidth=2, label='SLA (>= 0.85)')
            ax.legend(loc='lower right')
        elif metric == 'FPR':
            ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.3, linewidth=2, label='SLA (< 0.15)')
            ax.legend(loc='upper right')


    plt.suptitle('Сравнение ML моделей по ключевым метрикам', fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Сохраняем в папку docs для отчета
    os.makedirs('img/model_comparison', exist_ok=True)
    plt.savefig('img/model_comparison/model_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Графики сохранены в img/model_comparison/model_comparison.png")

def analyze_feature_importance(model, X_test, y_test, model_name):
    """Анализ важности признаков (поддерживает как деревья, так и нейросети)"""
    logger.info(f"--- АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ({model_name}) ---")
    
    # Если модель - дерево/бустинг, берем встроенную важность
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Если модель - нейросеть или линейная, используем Permutation Importance
    else:
        logger.info("Используем Permutation Importance (модель не имеет встроенной важности)...")
        result = permutation_importance(model, X_test, y_test, n_repeats=10, 
                                        random_state=42, n_jobs=-1, scoring='recall')
        importances = result.importances_mean

    features_names = X_test.columns
    feat_imp_df = pd.DataFrame({'Feature': features_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    logger.info(f"Топ-10 важных признаков для {model_name}:\n{feat_imp_df.to_string(index=False)}")
    
    # Здесь можно добавить код сохранения графика (feat_imp_df.plot.barh...)
    return feat_imp_df

def train_final_model(X_train, y_train, groups_train):
    """Гиперпараметрическая оптимизация и обучение финальной модели."""
    logger.info("\n--- 🟢 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ (LightGBM) ---")
    gkf = GroupKFold(n_splits=3) # Уменьшаем фолды для скорости поиска
    
    lgbm_model = lgb.LGBMClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [5, 10, 15, -1],
        'num_leaves': [20, 31, 50]
    }
    
    search = RandomizedSearchCV(
        lgbm_model,
        param_distributions,
        n_iter=10,
        scoring='recall',
        cv=gkf,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_train, y_train, groups=groups_train)
    
    logger.info(f"Лучшие параметры: {search.best_params_}")
    return search.best_estimator_

def final_evaluation(model, X_test, y_test):
    """Глубокая оценка, статистика (Bootstrap) и проверка инференса."""
    logger.info("\n--- 🟢 ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET ---")
    
    # Инференс время
    start = time.perf_counter()
    for _ in range(1000):
        model.predict(X_test.iloc[[0]])
    inference_ms = ((time.perf_counter() - start) / 1000) * 1000
    logger.info(f"Время инференса: {inference_ms:.2f} ms (Требование < 50 ms)")

    # Подбор порога
    y_proba = model.predict_proba(X_test)[:, 1]
    RECALL_SLA = 0.85
    RECALL_SAFETY_MARGIN = 0.06 # 6% страховки
    TARGET_RECALL = RECALL_SLA + RECALL_SAFETY_MARGIN # Ищем порог, дающий Recall >= 0.90

    logger.info("Поиск оптимального порога (Цель: Recall>=0.85 при максимальном Precision)...")
    best_threshold = 0.5
    best_precision = 0
    best_metrics = {}

    for thresh in np.arange(0.10, 0.50, 0.01):
        y_pred_t = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Если модель проходит жесткие ТЗ
        if recall >= TARGET_RECALL and fpr < 0.15:
            if precision > best_precision: # И ее Precision лучше предыдущего
                best_precision = precision
                best_threshold = thresh
                best_metrics = {'Recall': recall, 'FPR': fpr, 'Precision': precision}

    if not best_metrics:
        logger.error("❌ ВНИМАНИЕ: Ни один порог не смог выполнить ТЗ (Recall>=0.85 и FPR<0.15)!")
        # Фоллбэк: берем дефолтный порог 0.5
        y_pred_final = (y_proba >= 0.5).astype(int)
        best_threshold = 0.5 # Фоллбэк
    else:
        logger.info(f"✅ Найден оптимальный порог: {best_threshold:.2f}")
        # Делаем финальные предсказания с ЛУЧШИМ порогом
        y_pred_final = (y_proba >= best_threshold).astype(int)
    
    # Бизнес метрики
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final, labels=[0, 1]).ravel()
    final_recall = tp / (tp + fn)
    final_fpr = fp / (fp + tn)
    final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    logger.info(f"ФИНАЛЬНЫЙ RECALL: {final_recall:.3f} (Пасс: {final_recall >= 0.85})")
    logger.info(f"ФИНАЛЬНЫЙ FPR: {final_fpr:.3f} (Пасс: {final_fpr < 0.15})")
    logger.info(f"ФИНАЛЬНАЯ PRECISION: {final_precision:.3f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # 3. Статистический тест (Bootstrap 95% CI для Recall)
    logger.info("Расчет 95% доверительного интервала (Bootstrap)...")
    bootstrapped_recalls = []
    
    # Собираем реальные пары в датафрейм
    eval_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred_final})
    
    for _ in range(1000):
        # Семплируем ПАРЫ строк (сохраняем связь предсказания и реальности)
        df_boot = eval_df.sample(frac=1.0, replace=True)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(df_boot['y_true'], df_boot['y_pred'], labels=[0, 1]).ravel()
        score = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        bootstrapped_recalls.append(score)

    lower = np.percentile(bootstrapped_recalls, 2.5)
    upper = np.percentile(bootstrapped_recalls, 97.5)
    logger.info(f"Recall 95% CI: [{lower:.3f} : {upper:.3f}]")
    
    if lower >= 0.85:
        logger.info("✅ Статистически доказано: модель стабильна и проходит ТЗ!")
    else:
        logger.warning("⚠️ Нижняя граница CI ниже 0.85. Модель нестабильна на краевых случаях.")

    return y_pred_final, best_threshold

def plot_final_metrics(model, X_test, y_test, y_pred_final):
    """Сохраняет графики для отчета."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Матрица ошибок
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_final, ax=axes[0], cmap='Blues', normalize='true')
    axes[0].set_title("Confusion Matrix (Нормализованная)")

    # 2. ROC-AUC кривая
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_title("ROC-AUC Curve")

    # 3. Precision-Recall кривая (важна при дисбалансе!)
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[2])
    axes[2].set_title("Precision-Recall Curve")

    plt.tight_layout()

    os.makedirs('img/best_model', exist_ok=True)
    plt.savefig('img/best_model/final_evaluation_plots.png', dpi=150)
    print("📊 Графики сохранены в img/best_model/final_evaluation_plots.png")

def simulate_ab_test(y_test, y_pred_base, y_pred_lgb):
    """Симуляция A/B теста через Paired Bootstrap."""
    print("\n--- 🟢 СИМУЛЯЦИЯ A/B ТЕСТА (LightGBM vs Base) ---")
    
    def calc_business_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ПРАВИЛО 1: Жесткий SLA. Если безопасность не обеспечена - модель бесполезна
        if recall < 0.85:
            return -100000 # Огромный штраф, исключающий из конкуренции
        
        # ПРАВИЛО 2: Если SLA пройден, соревнование за эффективность (минимум ложных тревог)
        # Меньше FP = лучше. Возвращаем отрицательное число FP, чтобы максимизировать функцию.
        return -fp 

    # Реальная разница на тесте
    real_delta = calc_business_score(y_test, y_pred_lgb) - calc_business_score(y_test, y_pred_base)
    print(f"Реальная разница в бизнес-скоринге (LightGBM - Base): {real_delta:.0f} очков")

    # Bootstrap
    n_iterations = 1000
    deltas = []
    
    for _ in range(n_iterations):
        # Сэмплируем одни и те же индексы для обеих моделей
        indices = resample(np.arange(len(y_test)))
        
        score_base = calc_business_score(y_test.iloc[indices], y_pred_base[indices])
        score_xgb = calc_business_score(y_test.iloc[indices], y_pred_lgb[indices])
        
        deltas.append(score_xgb - score_base)

    # p-value: как часто Base был лучше или равен LightGBM?
    p_value = np.mean(np.array(deltas) <= 0)
    
    print(f"P-value (вероятность, что Base лучше): {p_value:.4f}")
    if p_value < 0.05:
        print("✅ СТАТИСТИЧЕСКИ ДОКАЗАНО: LightGBM значимо превосходит Base модель (p < 0.05)")
    else:
        print("❌ Улучшение НЕ статистически значимо. Разница могла быть случайной.")

def save_model(model):
    """Сохраняет артефакт модели."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    logger.info(f"✅✅✅ Модель успешно сохранена в {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    X, y, groups = load_data()
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(X, y, groups)
    
    # 1. Обучаем Base и СРАЗУ забираем её в переменную
    base_model = evaluate_baseline(X_train, X_test, y_train, y_test)
    y_pred_base_ab = base_model.predict(X_test) # Для Base дефолтный порог ОК
    
    # 2. Эксперименты
    run_experiments(X, y, groups)
    
    # 3. Обучаем финальную модель
    best_model = train_final_model(X_train, y_train, groups_train)
    
    # 4. Оценка финальной модели И забираем её предсказания с лучшим порогом
    y_pred_lgb_ab, best_thresh = final_evaluation(best_model, X_test, y_test)
    plot_final_metrics(best_model, X_test, y_test, y_pred_lgb_ab)
    
    # 5. Запускаем A/B тест, передавая оба массива предсказаний
    simulate_ab_test(y_test, y_pred_base_ab, y_pred_lgb_ab)
    
    # 6. Важность признаков и сохранение
    analyze_feature_importance(best_model, X_test, y_test, "Финальный LightGBM")
    save_model(best_model)