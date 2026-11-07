import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from streamlit.testing.v1 import AppTest

notebook = json.loads(Path('HW3_MMDA.ipynb').read_text(encoding='utf-8'))
scope = {'np': np, 'BaseEstimator': BaseEstimator, 'ClassifierMixin': ClassifierMixin}
exec(''.join(notebook['cells'][7]['source']), scope)
LogReg = scope['LogReg']
X = np.array([[-2.], [-1.], [1.], [2.]])
y = np.array([0, 0, 1, 1])
model = LogReg(eta=.1, max_iter=300).fit(X, y)
assert np.array_equal(model.predict(X), y)
assert np.allclose(model.predict_proba(X).sum(axis=1), 1)
assert np.isclose(model.loss_history[0], np.log(2))
assert len(model.loss_history) == model.n_iter_
initial = np.array([.3, -.5])
LogReg(w0=initial, max_iter=1).fit(X, y)
assert np.array_equal(initial, [.3, -.5])
a = LogReg(gd_type='stochastic', batch_size=2).fit(X, y)
b = LogReg(gd_type='stochastic', batch_size=2).fit(X, y)
assert np.array_equal(a.w, b.w)
assert np.isfinite(model.sigmoid(np.array([-1e6, 1e6]))).all()
model.w = np.array([0., 1e6])
assert np.isfinite(model.calc_loss(np.c_[np.ones(4), X], 1-y))
print('LogReg checks passed')

app = AppTest.from_file('wine_app/app.py', default_timeout=60).run()
assert not app.exception, app.exception
assert len(app.tabs) == 4
for threshold in [0., 1., .3]:
    app.slider[0].set_value(threshold).run()
    assert not app.exception, app.exception
print('Streamlit: four tabs and threshold changes passed')

# Проверяем чистую функцию обработки CSV, не запуская UI повторно.
tree = ast.parse(Path('wine_app/app.py').read_text(encoding='utf-8'))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'batch_predict')
namespace = {'pd': pd, 'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'), namespace)
predict = namespace['batch_predict']
pipeline = joblib.load('wine_app/model.joblib')
data = pd.read_csv('wine_app/test.csv').drop(columns='target').head()
features = list(pipeline.feature_names_in_)
result = predict(data[data.columns[::-1]], pipeline, features, .3)
assert np.allclose(result.probability, pipeline.predict_proba(data)[:, 1])
for bad in [data.drop(columns=features[0]), data.assign(alcohol=np.nan), data.assign(is_red=2), data.iloc[:0]]:
    try:
        predict(bad, pipeline, features, .3)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid CSV accepted')
print('Batch CSV validation passed')
