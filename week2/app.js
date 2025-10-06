// app.js — Final Fixed Version for Titanic Classifier (TensorFlow.js, tfjs-vis, PapaParse)

// Global state
let trainData = null, testData = null;
let preprocessedTrain = null, preprocessedTest = null;
let model = null, validation = null;
let validationPredsArr = null, testPredsArr = null;

// Helpers
const $ = id => document.getElementById(id);
const setDisabled = (id, state) => { const el = $(id); if (el) el.disabled = state; };

// Ensure TF backend ready (WebGL if available)
tf.setBackend('webgl').then(() => console.log('✅ TensorFlow backend:', tf.getBackend()));

// DOM ready
document.addEventListener('DOMContentLoaded', () => {
  $('load-data-btn').addEventListener('click', loadData);
  $('inspect-btn').addEventListener('click', inspectData);
  $('preprocess-btn').addEventListener('click', preprocessData);
  $('create-model-btn').addEventListener('click', createModel);
  $('train-btn').addEventListener('click', trainModel);
  $('threshold-slider').addEventListener('input', () => {
    $('threshold-value').textContent = parseFloat($('threshold-slider').value).toFixed(2);
    updateMetrics();
  });
  $('predict-btn').addEventListener('click', predictTest);
  $('export-btn').addEventListener('click', exportResults);
  $('save-model-btn').addEventListener('click', async () => {
    if (!model) return alert('No model to save.');
    await model.save('downloads://titanic-tfjs');
  });
});

// ---------------- CSV Loader ----------------
function parseFileWithPapa(file) {
  return new Promise((resolve, reject) => {
    if (!file) return reject(new Error('No file provided'));
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: h => h.trim(),
      complete: results => resolve(results.data),
      error: err => reject(err)
    });
  });
}

async function loadData() {
  const trainFile = $('train-file').files[0];
  const testFile = $('test-file').files[0];
  const status = $('data-status');
  if (!trainFile || !testFile) return alert('Please select both train.csv and test.csv files.');

  setDisabled('load-data-btn', true);
  status.textContent = 'Parsing CSVs... (PapaParse)';

  try {
    const [t, te] = await Promise.all([parseFileWithPapa(trainFile), parseFileWithPapa(testFile)]);
    const normalize = arr => arr.map(r => {
      const o = {};
      Object.keys(r).forEach(k => {
        const v = r[k];
        o[k] = (v === '' || v === undefined) ? null : v;
      });
      return o;
    });
    trainData = normalize(t);
    testData = normalize(te);

    status.textContent = `✅ Loaded Train: ${trainData.length} rows, Test: ${testData.length} rows`;
    setDisabled('inspect-btn', false);
    setDisabled('preprocess-btn', false);
  } catch (err) {
    console.error(err);
    alert('Error parsing CSV: ' + err.message);
    status.textContent = '❌ CSV load failed';
  } finally {
    setDisabled('load-data-btn', false);
  }
}

// ---------------- Inspect Data ----------------
function inspectData() {
  if (!trainData) return alert('Load data first.');

  $('data-preview').innerHTML = makeTable(trainData.slice(0, 8));
  const cols = Object.keys(trainData[0]);
  const missing = cols.map(c => {
    const miss = trainData.filter(r => r[c] === null).length;
    return { column: c, missing: ((miss / trainData.length) * 100).toFixed(2) + '%' };
  });
  $('data-stats').innerHTML = '<pre>' + JSON.stringify(missing, null, 2) + '</pre>';

  // Visuals
  const survBySex = {};
  trainData.forEach(r => {
    const s = r.Sex ?? 'Unknown';
    survBySex[s] = survBySex[s] || { surv: 0, total: 0 };
    if (r.Survived === 1) survBySex[s].surv++;
    survBySex[s].total++;
  });
  const sexData = Object.keys(survBySex).map(k => ({ x: k, y: (survBySex[k].surv / survBySex[k].total) * 100 }));
  tfvis.render.barchart({ name: 'Survival by Sex', tab: 'Inspect' }, sexData);

  const survByPclass = {};
  trainData.forEach(r => {
    const c = r.Pclass ?? 'Unknown';
    survByPclass[c] = survByPclass[c] || { surv: 0, total: 0 };
    if (r.Survived === 1) survByPclass[c].surv++;
    survByPclass[c].total++;
  });
  const classData = Object.keys(survByPclass).map(k => ({ x: 'Class ' + k, y: (survByPclass[k].surv / survByPclass[k].total) * 100 }));
  tfvis.render.barchart({ name: 'Survival by Pclass', tab: 'Inspect' }, classData);

  const ageSurv = trainData.filter(r => r.Survived === 1 && r.Age != null).map(r => r.Age);
  const ageDead = trainData.filter(r => r.Survived === 0 && r.Age != null).map(r => r.Age);
  tfvis.render.histogram({ name: 'Age Distribution by Survival', tab: 'Inspect' }, [
    { values: ageSurv, series: ['Survived'] },
    { values: ageDead, series: ['Died'] }
  ], { bins: 15 });
}

function makeTable(rows) {
  const cols = Object.keys(rows[0]);
  let html = '<table><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
  rows.forEach(r => html += '<tr>' + cols.map(c => `<td>${r[c] ?? ''}</td>`).join('') + '</tr>');
  return html + '</table>';
}

// ---------------- Preprocessing ----------------
function preprocessData() {
  if (!trainData) return alert('Load train data first.');
  const addFamily = $('add-family-features').checked;
  $('preprocessing-output').textContent = 'Processing...';

  const median = arr => { const s = arr.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)] || 0; };
  const std = arr => { const m = arr.reduce((a, b) => a + b, 0) / arr.length; return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length) || 1; };
  const mode = arr => { const f = {}; arr.forEach(v => { if (v != null) f[v] = (f[v] || 0) + 1; }); return Object.keys(f).sort((a, b) => f[b] - f[a])[0]; };

  const ageVals = trainData.map(r => r.Age).filter(v => v != null);
  const fareVals = trainData.map(r => r.Fare).filter(v => v != null);
  const ageMed = median(ageVals), ageStd = std(ageVals);
  const fareMed = median(fareVals), fareStd = std(fareVals);
  const embMode = mode(trainData.map(r => r.Embarked));

  const cats = {
    Pclass: [...new Set(trainData.map(r => r.Pclass ?? 'NA'))],
    Sex: [...new Set(trainData.map(r => r.Sex ?? 'NA'))],
    Embarked: [...new Set(trainData.map(r => r.Embarked ?? 'NA'))]
  };

  function extract(row) {
    const age = (row.Age ?? ageMed - ageMed) / ageStd;
    const fare = (row.Fare ?? fareMed - fareMed) / fareStd;
    const sibsp = row.SibSp ?? 0, parch = row.Parch ?? 0;
    const features = [age, fare, sibsp, parch];
    if (addFamily) {
      const fs = sibsp + parch + 1;
      features.push(fs, fs === 1 ? 1 : 0);
    }
    cats.Pclass.forEach(v => features.push(row.Pclass === v ? 1 : 0));
    cats.Sex.forEach(v => features.push(row.Sex === v ? 1 : 0));
    cats.Embarked.forEach(v => features.push(row.Embarked === v ? 1 : 0));
    return features;
  }

  preprocessedTrain = {
    featuresArr: trainData.map(extract),
    labelsArr: trainData.map(r => (r.Survived === 1 ? 1 : 0))
  };
  preprocessedTest = {
    featuresArr: testData.map(extract),
    passengerIds: testData.map(r => r.PassengerId)
  };

  $('preprocessing-output').textContent =
    `✅ Done. ${preprocessedTrain.featuresArr.length} samples, ${preprocessedTrain.featuresArr[0].length} features.`;
  setDisabled('create-model-btn', false);
}

// ---------------- Model ----------------
function createModel() {
  const inputDim = preprocessedTrain.featuresArr[0].length;
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [inputDim] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  $('model-summary').textContent = `Model: Dense(8,relu) → Dense(1,sigmoid) (${model.countParams()} params)`;
  setDisabled('train-btn', false);
}

// ---------------- Train ----------------
async function trainModel() {
  if (!model || !preprocessedTrain) return alert('Prepare model & data first.');

  const { xTrain, yTrain, xVal, yVal } = stratifiedSplit(preprocessedTrain.featuresArr, preprocessedTrain.labelsArr, 0.2, 123);
  const fLen = preprocessedTrain.featuresArr[0].length;

  const clean = arr => arr.filter(r => Array.isArray(r) && r.length === fLen);
  const trainX = tf.tensor2d(clean(xTrain));
  const trainY = tf.tensor2d(yTrain.slice(0, trainX.shape[0]), [trainX.shape[0], 1]);
  const valX = tf.tensor2d(clean(xVal));
  const valY = tf.tensor2d(yVal.slice(0, valX.shape[0]), [valX.shape[0], 1]);
  validation = { featuresTensor: valX, labelsTensor: valY };

  const fitCallbacks = tfvis.show.fitCallbacks({ name: 'Training', tab: 'Training' },
    ['loss', 'val_loss', 'acc', 'val_acc'], { callbacks: ['onEpochEnd'] });
  const earlyStop = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 });

  $('training-status').textContent = 'Training (≈10s)…';
  await tf.nextFrame();

  const history = await model.fit(trainX, trainY, {
    epochs: 30, batchSize: 16,
    validationData: [valX, valY],
    callbacks: [fitCallbacks, earlyStop, {
      onEpochEnd: async (e, logs) => {
        $('training-status').textContent =
          `Epoch ${e + 1}: loss=${logs.loss.toFixed(4)}, val_acc=${(logs.val_acc || 0).toFixed(4)}`;
        await tf.nextFrame();
      }
    }]
  });

  $('training-status').textContent = '✅ Training finished.';
  const preds = model.predict(valX);
  validationPredsArr = preds.dataSync();
  validation.labels = valY.dataSync();
  preds.dispose();

  computeROCAndPlot(validation.labels, validationPredsArr);
  setDisabled('threshold-slider', false);
  setDisabled('predict-btn', false);
  setDisabled('save-model-btn', false);

  trainX.dispose(); trainY.dispose();
}

// ---------------- ROC + Metrics ----------------
function computeROCAndPlot(yTrue, yProb) {
  const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
  const roc = thresholds.map(t => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    yTrue.forEach((yt, i) => {
      const yp = yProb[i] >= t ? 1 : 0;
      if (yt === 1 && yp === 1) tp++;
      if (yt === 0 && yp === 1) fp++;
      if (yt === 0 && yp === 0) tn++;
      if (yt === 1 && yp === 0) fn++;
    });
    return { x: fp / (fp + tn + 1e-6), y: tp / (tp + fn + 1e-6) };
  });

  tfvis.render.linechart({ name: 'ROC Curve', tab: 'Evaluation' },
    { values: roc }, { xLabel: 'FPR', yLabel: 'TPR' });
  updateMetrics();
}

function updateMetrics() {
  if (!validation || !validationPredsArr) return;
  const t = parseFloat($('threshold-slider').value);
  let tp = 0, tn = 0, fp = 0, fn = 0;
  validation.labels.forEach((yt, i) => {
    const yp = validationPredsArr[i] >= t ? 1 : 0;
    if (yt === 1 && yp === 1) tp++;
    if (yt === 0 && yp === 1) fp++;
    if (yt === 0 && yp === 0) tn++;
    if (yt === 1 && yp === 0) fn++;
  });
  const prec = tp / (tp + fp + 1e-6);
  const rec = tp / (tp + fn + 1e-6);
  const f1 = 2 * prec * rec / (prec + rec + 1e-6);
  $('confusion-matrix').innerHTML = `
    <table border="1"><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
    <tr><th>True 0</th><td>${tn}</td><td>${fp}</td></tr>
    <tr><th>True 1</th><td>${fn}</td><td>${tp}</td></tr></table>`;
  $('performance-metrics').innerHTML =
    `<p>Precision: ${(prec * 100).toFixed(1)}% | Recall: ${(rec * 100).toFixed(1)}% | F1: ${(f1 * 100).toFixed(1)}%</p>`;
}

// ---------------- Predict + Export ----------------
async function predictTest() {
  const testX = tf.tensor2d(preprocessedTest.featuresArr);
  const preds = model.predict(testX);
  testPredsArr = preds.dataSync();
  $('prediction-output').textContent = 'Sample probs: ' + testPredsArr.slice(0, 10).map(p => p.toFixed(3)).join(', ');
  setDisabled('export-btn', false);
  preds.dispose(); testX.dispose();
}

function exportResults() {
  let csv = 'PassengerId,Survived\n';
  for (let i = 0; i < testPredsArr.length; i++) {
    const id = preprocessedTest.passengerIds[i] ?? i + 1;
    csv += `${id},${testPredsArr[i] >= 0.5 ? 1 : 0}\n`;
  }
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'submission.csv';
  a.click();
}

// ---------------- Stratified Split ----------------
function stratifiedSplit(X, y, valRatio = 0.2, seed = 42) {
  const groups = {};
  y.forEach((lbl, i) => { groups[lbl] = groups[lbl] || []; groups[lbl].push(i); });
  const rand = mulberry32(seed);
  const trainIdx = [], valIdx = [];
  Object.values(groups).forEach(indices => {
    const arr = indices.slice();
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    const cut = Math.floor(arr.length * (1 - valRatio));
    trainIdx.push(...arr.slice(0, cut));
    valIdx.push(...arr.slice(cut));
  });
  return {
    xTrain: trainIdx.map(i => X[i]),
    yTrain: trainIdx.map(i => y[i]),
    xVal: valIdx.map(i => X[i]),
    yVal: valIdx.map(i => y[i])
  };
}
function mulberry32(a) { return function () { let t = a += 0x6D2B79F5; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
