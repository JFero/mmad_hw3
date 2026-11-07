from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Винная подборка', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def batch_predict(frame, model, features, threshold):
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError('В CSV не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('CSV не содержит строк с винами.')
    try:
        inputs = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Все признаки должны быть числами.') from exc
    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError('Удалите пропуски и бесконечные значения из признаков.')
    if not inputs['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен быть равен 0 (белое) или 1 (красное).')
    result = frame.copy()
    result['probability'] = model.predict_proba(inputs)[:, 1]
    result['recommend'] = (result['probability'] >= threshold).astype(int)
    return result


model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X_test, y_test = test[features], test['target']
prob = test_proba(model, X_test)
config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
st.title('🍷 Винная подборка')
st.caption('Оценка вероятности экспертного качества ≥ 7 по лабораторным характеристикам.')
threshold = st.sidebar.slider('Порог рекомендации', 0., 1., float(config['threshold']),
                              step=.001, format='%.3f', key='threshold')
st.sidebar.caption('Один порог используется на всех вкладках.')
single, evaluation, batch, coefficients = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with single:
    st.subheader('Характеристики вина')
    values = {}
    columns = st.columns(3)
    for i, name in enumerate(features):
        with columns[i % 3]:
            if name == 'is_red':
                values[name] = int(st.toggle('Красное вино', value=bool(X_test[name].median())))
            else:
                values[name] = st.number_input(name, min_value=float(X_test[name].min()),
                    max_value=float(X_test[name].max()), value=float(X_test[name].median()),
                    format='%.5f')
    single_prob = float(model.predict_proba(pd.DataFrame([values], columns=features))[0, 1])
    st.metric('Вероятность хорошего вина', f'{single_prob:.1%}')
    st.success('Рекомендовать в подборку') if single_prob >= threshold else st.info('Не включать в подборку')

with evaluation:
    pred = prob >= threshold
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    for col, name, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        col.metric(name, f'{value:.3f}')
    st.write('При повышении порога подборка уменьшается: обычно растёт доля хороших вин, '
             'но больше хороших образцов остаётся за бортом. При снижении порога полнота растёт, '
             'но в подборку попадает больше посредственных вин. Precision не обязан меняться монотонно.')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Other', 'Good']).plot(ax=axes[0], colorbar=False)
    metrics.RocCurveDisplay.from_predictions(y_test, prob, ax=axes[1])
    tn, fp, fn, tp = cm.ravel()
    axes[1].scatter(fp/(fp+tn), recall, s=70, label=f't={threshold:.3f}')
    axes[1].plot([0, 1], [0, 1], '--', color='gray'); axes[1].legend()
    metrics.PrecisionRecallDisplay.from_predictions(y_test, prob, ax=axes[2])
    axes[2].scatter(recall, precision, s=70, label=f't={threshold:.3f}')
    axes[2].axhline(y_test.mean(), ls='--', color='gray', label='Prevalence'); axes[2].legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption('Кривые рассчитаны по вероятностям и не меняются при движении порога. '
               'Тестовые данные служат демонстрацией; подбирать рабочий порог по ним нельзя.')

with batch:
    st.write('Загрузите CSV с признаками; target не требуется. Столбцы: ' + ', '.join(features))
    uploaded = st.file_uploader('CSV с винами', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            result = batch_predict(frame, model, features, threshold)
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(str(exc))
        else:
            st.dataframe(result)
            st.download_button('Скачать результат', result.to_csv(index=False).encode('utf-8-sig'),
                               file_name='wine_predictions.csv', mime='text/csv')

with coefficients:
    weights = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    weights.plot.barh(ax=ax, color=np.where(weights >= 0, '#24856b', '#b9565a'))
    ax.axvline(0, color='black', lw=.8)
    ax.set_xlabel('Коэффициент для стандартизированного признака')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(weights.nlargest(3).index) + '.')
    st.write('Сильнее всего против: ' + ', '.join(weights.nsmallest(3).index) + '.')
    st.caption('Коэффициент — изменение логарифма шансов при увеличении признака на одно '
               'стандартное отклонение, если остальные признаки фиксированы. Это ассоциация, не причинность.')
