// app.js — Titanic Classifier with Enhanced Visualizations

let trainData = [], testData = [];
let xTrain, yTrain, xVal, yVal, testX;
let featureNames = [];
let model;
let valPreds = [], valLabels = [];

const $ = id => document.getElementById(id);
const setDisabled = (id, state) => { $(id).disabled = state; };

// --- CSV Loader ---
$('load-data-btn').addEventListener('click', async () => {
  try {
    trainData = await loadCSV($('trainFile'));
    testData = await loadCSV($('testFile'));
    $('data-status').textContent = `Train: ${trainData.length} rows, Test: ${testData.length} rows`;
    setDisabled('inspect-btn', false);
  } catch (err) { alert("CSV load error: " + err); }
});

function loadCSV(input) {
  return new Promise((resolve, reject) => {
    if (!input.files[0]) return reject("No file chosen");
    Papa.parse(input.files[0], {
      header: true, dynamicTyping: true, skipEmptyLines: true,
      complete: res => resolve(res.data), error: err => reject(err)
    });
  });
}

// --- Inspect Step ---
$('inspect-btn').addEventListener('click', () => {
  $('data-preview').textContent = JSON.stringify(trainData.slice(0, 5), null, 2);

  // Missing values
  const cols = Object.keys(trainData[0]);
  const missing = cols.map(c => ({
    col: c,
    missing: trainData.filter(r => r[c] === null || r[c] === "").length
  }));
  $('data-stats').textContent = JSON.stringify(missing, null, 2);

  // Survival by Sex
  const survBySex = {};
  trainData.forEach(r => {
    if (r.Survived != null) {
      survBySex[r.Sex] = survBySex[r.Sex] || { surv: 0, total: 0 };
      survBySex[r.Sex].total++;
      if (r.Survived === 1) survBySex[r.Sex].surv++;
    }
  });
  const sexValues = Object.keys(survBySex).map(k => ({
    x: k,
    y: survBySex[k].surv / survBySex[k].total
  }));
  tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Inspect' }, sexValues);

  // Survival by Pclass
  const survByPclass = {};
  trainData.forEach(r => {
    const cls = r.Pclass;
    survByPclass[cls] = survByPclass[cls] || { surv: 0, total: 0 };
    survByPclass[cls].total++;
    if (r.Survived === 1) survByPclass[cls].surv++;
  });
  const pclassValues = Object.keys(survByPclass).map(k => ({
    x: "Class " + k,
    y: survByPclass[k].surv / survByPclass[k].total
  }));
  tfvis.render.barchart({ name: 'Survival Rate by Pclass', tab: 'Inspect' }, pclassValues);

  // Age distribution by Survival
  const ageSurvived = trainData.filter(r => r.Survived === 1 && r.Age != null).map(r => r.Age);
  const ageDied = trainData.filter(r => r.Survived === 0 && r.Age != null).map(r => r.Age);
  const histData = [
    { values: ageSurvived, bins: 15, color: 'green', label: 'Survived' },
    { values: ageDied, bins: 15, color: 'red', label: 'Died' }
  ];
  tfvis.render.histogram({ name: 'Age Distribution by Survival', tab: 'Inspect' }, histData);

  setDisabled('preprocess-btn', false);
});

// --- Preprocess ---
$('preprocess-btn').addEventListener('click', () => {
  const addFamily = $('toggle-family').checked;

  const imputeMedian = (arr, key) => {
    const vals = arr.map(r => r[key]).filter(v => v != null);
    vals.sort((a, b) => a - b);
    return vals[Math.floor(vals.length / 2)];
  };
  const imputeMode = (arr, key) => {
    const counts = {};
    arr.forEach(r => { counts[r[key]] = (counts[r[key]] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  };

  const ageMed = imputeMedian(trainData, 'Age');
  const embMode = imputeMode(trainData, 'Embarked');
  trainData.forEach(r => { if (!r.Age) r.Age = ageMed; if (!r.Embarked) r.Embarked = embMode; });
  testData.forEach(r => { if (!r.Age) r.Age = ageMed; if (!r.Embarked) r.Embarked = embMode; });

  featureNames = ['Age', 'Fare'];
  if (addFamily) featureNames.push('FamilySize', 'IsAlone');
  ['Pclass', 'Sex', 'Embarked'].forEach(cat => {
    [...new Set(trainData.map(r => r[cat]))].forEach(u => featureNames.push(`${cat}_${u}`));
  });

  const proc = rows => rows.map(r => {
    let feats = [];
    feats.push((r.Age - ageMed) / ageMed);
    feats.push((r.Fare - (r.Fare || 0)) / ((r.Fare || 1)));
    if (addFamily) {
      const fs = r.SibSp + r.Parch + 1;
      feats.push(fs, fs === 1 ? 1 : 0);
    }
    ['Pclass', 'Sex', 'Embarked'].forEach(cat => {
      [...new Set(trainData.map(rr => rr[cat]))].forEach(u => feats.push(r[cat] === u ? 1 : 0));
    });
    return feats;
  });

  const X = proc(trainData), Y = trainData.map(r => r.Survived);
  const N = X.length, split = Math.floor(N * 0.8);
  xTrain = tf.tensor2d(X.slice(0, split));
  yTrain = tf.tensor2d(Y.slice(0, split), [split, 1]);
  xVal = tf.tensor2d(X.slice(split));
  yVal = tf.tensor2d(Y.slice(split), [N - split, 1]);
  testX = tf.tensor2d(proc(testData));

  $('preprocessing-output').textContent = `Features: ${featureNames.join(", ")}\nTrain shape: ${xTrain.shape}`;

  // Correlation heatmap
  const corr = tf.matMul(xTrain.transpose(), xTrain).div(xTrain.shape[0]).arraySync();
  tfvis.render.heatmap(
    { name: 'Feature Correlation Heatmap', tab: 'Preprocess' },
    { values: corr, xTickLabels: featureNames, yTickLabels: featureNames }
  );

  setDisabled('create-model-btn', false);
});

// --- Model ---
$('create-model-btn').addEventListener('click', () => {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [featureNames.length] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  $('model-summary').textContent = "Model created with Dense(16, relu) → Dense(1, sigmoid).";
  setDisabled('train-btn', false);
});

// --- Train ---
$('train-btn').addEventListener('click', async () => {
  await model.fit(xTrain, yTrain, {
    epochs: 50, batchSize: 32, validationData: [xVal, yVal],
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Progress', tab: 'Training' },
      ['loss', 'val_loss', 'acc', 'val_acc'], { callbacks: ['onEpochEnd'] })
  });

  valPreds = model.predict(xVal).arraySync().map(r => r[0]);
  valLabels = yVal.arraySync().map(r => r[0]);

  computeROC(valLabels, valPreds);
  updateMetrics(0.5);

  setDisabled('predict-btn', false);
  setDisabled('export-btn', false);
  setDisabled('save-model-btn', false);
});

// --- ROC Curve ---
function computeROC(yTrue, yProb) {
  let thresholds = [];
  for (let t = 0; t <= 1; t += 0.05) thresholds.push(t);

  const roc = thresholds.map(t => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    yTrue.forEach((yt, i) => {
      const yp = yProb[i] >= t ? 1 : 0;
      if (yt === 1 && yp === 1) tp++;
      if (yt === 0 && yp === 1) fp++;
      if (yt === 0 && yp === 0) tn++;
      if (yt === 1 && yp === 0) fn++;
    });
    const tpr = tp / (tp + fn + 1e-6);
    const fpr = fp / (fp + tn + 1e-6);
    return { x: fpr, y: tpr };
  });
  tfvis.render.scatterplot({ name: 'ROC Curve', tab: 'Metrics' }, { values: roc }, { xLabel: 'FPR', yLabel: 'TPR' });
}

// --- Threshold Metrics ---
$('threshold-slider').addEventListener('input', e => {
  const t = parseFloat(e.target.value);
  $('threshold-value').textContent = t.toFixed(2);
  updateMetrics(t);
});

function updateMetrics(thresh) {
  if (valPreds.length === 0) return;
  let tp = 0, fp = 0, tn = 0, fn = 0;
  valLabels.forEach((yt, i) => {
    const yp = valPreds[i] >= thresh ? 1 : 0;
    if (yt === 1 && yp === 1) tp++;
    if (yt === 0 && yp === 1) fp++;
    if (yt === 0 && yp === 0) tn++;
    if (yt === 1 && yp === 0) fn++;
  });
  const prec = tp / (tp + fp + 1e-6);
  const rec = tp / (tp + fn + 1e-6);
  const f1 = 2 * prec * rec / (prec + rec + 1e-6);

  $('metrics-output').innerHTML = `
    <div class="metric-card">Precision: ${(prec*100).toFixed(1)}%</div>
    <div class="metric-card">Recall: ${(rec*100).toFixed(1)}%</div>
    <div class="metric-card">F1: ${(f1*100).toFixed(1)}%</div>
  `;

  $('confusion-matrix').innerHTML = `
    <table border="1" cellpadding="5">
      <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
      <tr><th>True 0</th><td>${tn}</td><td>${fp}</td></tr>
      <tr><th>True 1</th><td>${fn}</td><td>${tp}</td></tr>
    </table>
  `;
}

// --- Predict & Export ---
$('predict-btn').addEventListener('click', () => {
  const preds = model.predict(testX).arraySync().map(r => r[0]);
  $('prediction-output').textContent = preds.slice(0, 10).map(p => p.toFixed(3)).join(", ") + "...";
});

$('export-btn').addEventListener('click', () => {
  const preds = model.predict(testX).arraySync().map(r => r[0]);
  let csv = "PassengerId,Survived\n";
  testData.forEach((r, i) => {
    csv += `${r.PassengerId},${preds[i] >= 0.5 ? 1 : 0}\n`;
  });
  downloadCSV(csv, "submission.csv");
});

$('save-model-btn').addEventListener('click', async () => {
  await model.save('downloads://titanic-tfjs');
});

function downloadCSV(csv, name) {
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}
