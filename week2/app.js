// app.js - Titanic classifier (browser-only, uses tfjs, tfjs-vis, PapaParse)

// Globals
let trainData = null;
let testData = null;
let preprocessedTrain = null; // {featuresArr, labelsArr}
let preprocessedTest = null;  // {featuresArr, passengerIds}
let model = null;
let validation = null;        // {featuresTensor, labelsTensor}
let validationPredsArr = null;
let testPredsArr = null;

// Quick DOM helpers
const $ = id => document.getElementById(id);
const setDisabled = (id, state) => { const el = $(id); if (el) el.disabled = state; };

// Ensure DOM ready
document.addEventListener('DOMContentLoaded', () => {
  // Wire UI buttons
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

// --------------------
// CSV loading (PapaParse) - returns Promise of array of objects
// --------------------
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

  if (!trainFile || !testFile) {
    alert('Please select both train.csv and test.csv files.');
    return;
  }

  setDisabled('load-data-btn', true);
  status.textContent = 'Parsing CSVs... (PapaParse)';

  try {
    // Parse in parallel
    const [t, te] = await Promise.all([
      parseFileWithPapa(trainFile),
      parseFileWithPapa(testFile)
    ]);

    // Convert empty strings to null to standardize
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

    status.textContent = `Loaded. Train rows: ${trainData.length}, Test rows: ${testData.length}`;
    setDisabled('inspect-btn', false);
    setDisabled('preprocess-btn', false);
  } catch (err) {
    console.error(err);
    status.textContent = 'Error loading CSVs: ' + (err.message || err);
    alert('Error parsing CSV files. See console for details.');
  } finally {
    setDisabled('load-data-btn', false);
  }
}

// --------------------
// Inspect data & simple viz
// --------------------
function inspectData() {
  if (!trainData || trainData.length === 0) return alert('Load train.csv first');

  // Preview first 8 rows as table
  const previewDiv = $('data-preview');
  previewDiv.innerHTML = '<h4>Preview (first 8 rows)</h4>' + makeTable(trainData.slice(0, 8));

  // Missing % per column
  const statsDiv = $('data-stats');
  const cols = Object.keys(trainData[0]);
  const missing = cols.map(c => {
    const miss = trainData.filter(r => r[c] === null).length;
    return { column: c, missing: (miss / trainData.length * 100).toFixed(2) + '%' };
  });
  statsDiv.innerHTML = '<h4>Missing %</h4><pre style="white-space:pre-wrap;">' + JSON.stringify(missing, null, 2) + '</pre>';

  // tfjs-vis charts
  // Survival rate by Sex
  const survivalBySex = {};
  trainData.forEach(r => {
    const s = r.Sex ?? 'unknown';
    survivalBySex[s] = survivalBySex[s] || { survived: 0, total: 0 };
    if (r.Survived === 1) survivalBySex[s].survived++;
    survivalBySex[s].total++;
  });
  const sexData = Object.keys(survivalBySex).map(k => ({ x: k, y: (survivalBySex[k].survived / survivalBySex[k].total) * 100 }));
  tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Inspect' }, sexData, { xLabel: 'Sex', yLabel: 'Survival %' });

  // Survival rate by Pclass
  const survivalByClass = {};
  trainData.forEach(r => {
    const p = String(r.Pclass ?? 'unknown');
    survivalByClass[p] = survivalByClass[p] || { survived: 0, total: 0 };
    if (r.Survived === 1) survivalByClass[p].survived++;
    survivalByClass[p].total++;
  });
  const classData = Object.keys(survivalByClass).map(k => ({ x: 'Class ' + k, y: (survivalByClass[k].survived / survivalByClass[k].total) * 100 }));
  tfvis.render.barchart({ name: 'Survival Rate by Pclass', tab: 'Inspect' }, classData);

  // Age hist by survival (use tfvis histogram)
  const ageSurv = trainData.filter(r => r.Survived === 1 && r.Age != null).map(r => r.Age);
  const ageDead = trainData.filter(r => r.Survived === 0 && r.Age != null).map(r => r.Age);
  tfvis.render.histogram({ name: 'Age distribution (survived vs died)', tab: 'Inspect' }, [{ values: ageSurv, series: ['Survived'] }, { values: ageDead, series: ['Died'] }], { bins: 15 });

  // Enable preprocess (already enabled on load, but ensure)
  setDisabled('preprocess-btn', false);
}

// small helper to build a table from objects
function makeTable(rows) {
  if (!rows || rows.length === 0) return '<div>No preview</div>';
  const cols = Object.keys(rows[0]);
  let html = '<table><thead><tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>' + cols.map(c => `<td>${escapeHtml(String(r[c] ?? ''))}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  return html;
}
function escapeHtml(s){ return s.replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }

// --------------------
// Preprocessing
// --------------------
function preprocessData() {
  if (!trainData || !testData) return alert('Load data first');

  const addFamily = $('add-family-features').checked;
  const outDiv = $('preprocessing-output');
  outDiv.textContent = 'Preprocessing...';

  // Helper stats from trainData
  const numeric = arr => arr.filter(v => v != null && !isNaN(v)).map(Number);
  const median = arr => { if (!arr.length) return 0; const s = arr.slice().sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; };
  const std = arr => { if (!arr.length) return 1; const m = arr.reduce((a,b)=>a+b,0)/arr.length; return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length) || 1; };
  const mode = arr => { const freq={}; arr.forEach(v=>{ if(v!=null) freq[v]= (freq[v]||0)+1;}); const e = Object.entries(freq); if(!e.length) return null; e.sort((a,b)=>b[1]-a[1]); return e[0][0]; };

  const ageVals = numeric(trainData.map(r=>r.Age));
  const fareVals = numeric(trainData.map(r=>r.Fare));
  const ageMedian = median(ageVals);
  const fareMedian = median(fareVals);
  const ageStd = std(ageVals);
  const fareStd = std(fareVals);
  const embarkedMode = mode(trainData.map(r=>r.Embarked));

  // Create one-hot categories from train set for consistent encoding
  const unique = (col) => Array.from(new Set(trainData.map(r => (r[col] == null ? '__NA__' : r[col]))));
  const pclassCats = unique('Pclass');
  const sexCats = unique('Sex');
  const embarkedCats = unique('Embarked');

  // Feature extraction function
  function extractFeatures(row) {
    // impute
    const age = (row.Age != null && !isNaN(row.Age)) ? Number(row.Age) : ageMedian;
    const fare = (row.Fare != null && !isNaN(row.Fare)) ? Number(row.Fare) : fareMedian;
    const sibsp = (row.SibSp != null && !isNaN(row.SibSp)) ? Number(row.SibSp) : 0;
    const parch = (row.Parch != null && !isNaN(row.Parch)) ? Number(row.Parch) : 0;
    const embarked = row.Embarked != null ? row.Embarked : embarkedMode;

    // standardized numeric (z-score)
    const ageStdNorm = (age - ageMedian) / (ageStd || 1);
    const fareStdNorm = (fare - fareMedian) / (fareStd || 1);

    const features = [ageStdNorm, fareStdNorm, sibsp, parch];

    if (addFamily) {
      const familySize = sibsp + parch + 1;
      const isAlone = familySize === 1 ? 1 : 0;
      features.push(familySize, isAlone);
    }

    // one-hot Pclass, Sex, Embarked using categories defined from training set
    pclassCats.forEach(cat => features.push((row.Pclass == cat) ? 1 : 0));
    sexCats.forEach(cat => features.push((row.Sex == cat) ? 1 : 0));
    embarkedCats.forEach(cat => features.push(( (row.Embarked == cat) ? 1 : 0)));
    return features;
  }

  // Build arrays
  const X_train = trainData.map(r => extractFeatures(r));
  const y_train = trainData.map(r => (r.Survived === 1 ? 1 : 0));
  const X_test = testData.map(r => extractFeatures(r));
  const ids = testData.map(r => r.PassengerId ?? null);

  preprocessedTrain = { featuresArr: X_train, labelsArr: y_train };
  preprocessedTest = { featuresArr: X_test, passengerIds: ids };

  outDiv.innerHTML = `<p>Preprocessing complete. Feature size: ${X_train[0].length}. Train samples: ${X_train.length}.</p>`;

  // Small correlation heatmap (approx) using tfjs-vis
  try {
    const tfX = tf.tensor2d(X_train);
    const corr = tf.matMul(tfX.transpose(), tfX).div(tfX.shape[0]).arraySync();
    tfX.dispose();
    tfvis.render.heatmap({ name: 'Feature Correlation', tab: 'Preprocess' }, { values: corr });
  } catch (e) { console.warn('Heatmap failed:', e); }

  // Enable later steps
  setDisabled('create-model-btn', false);
  setDisabled('train-btn', false);
}

// --------------------
// Model creation
// --------------------
function createModel() {
  if (!preprocessedTrain) return alert('Preprocess data first');

  const inputDim = preprocessedTrain.featuresArr[0].length;
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  $('model-summary').innerHTML = `<p>Model: Dense(16,relu) → Dense(1,sigmoid). Params: ${model.countParams()}</p>`;
  setDisabled('train-btn', false);
}

// --------------------
// Stratified split helper
// --------------------
function stratifiedSplit(featuresArr, labelsArr, valRatio=0.2, seed=42) {
  // group indices by label
  const groups = {};
  labelsArr.forEach((lbl, i) => { groups[lbl] = groups[lbl] || []; groups[lbl].push(i); });

  // simple seeded shuffle
  function mulberry32(a){ return function(){ var t = a += 0x6D2B79F5; t = Math.imul(t ^ t>>>15, t | 1); t ^= t + Math.imul(t ^ t>>>7, t | 61); return ((t ^ t>>>14) >>> 0) / 4294967296; }; }
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

  const xTrain = trainIdx.map(i => featuresArr[i]);
  const yTrain = trainIdx.map(i => labelsArr[i]);
  const xVal = valIdx.map(i => featuresArr[i]);
  const yVal = valIdx.map(i => labelsArr[i]);

  return { xTrain, yTrain, xVal, yVal };
}

// --------------------
// Training
// --------------------
async function trainModel() {
  if (!model || !preprocessedTrain) return alert('Prepare model and data first');

  setDisabled('train-btn', true);
  $('training-status').textContent = 'Preparing training split...';

  // Stratified split
  const { xTrain, yTrain, xVal, yVal } = stratifiedSplit(preprocessedTrain.featuresArr, preprocessedTrain.labelsArr, 0.2, 1234);

  const trainX = tf.tensor2d(xTrain);
  const trainY = tf.tensor2d(yTrain, [yTrain.length, 1]);
  const valX = tf.tensor2d(xVal);
  const valY = tf.tensor2d(yVal, [yVal.length, 1]);

  validation = { featuresTensor: valX, labelsTensor: valY };

  // Combine tfjs-vis callbacks and early stopping
  const visCallbacks = tfvis.show.fitCallbacks(
    { name: 'Training Performance', tab: 'Training' },
    ['loss', 'val_loss', 'acc', 'val_acc'],
    { callbacks: ['onEpochEnd'] }
  );
  const earlyStopping = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 });

  $('training-status').textContent = 'Training... (this can take a minute)';

  await model.fit(trainX, trainY, {
    epochs: 50,
    batchSize: 32,
    validationData: [valX, valY],
    callbacks: [visCallbacks, earlyStopping, {
      onEpochEnd: (epoch, logs) => {
        $('training-status').textContent = `Epoch ${epoch+1}/50 — loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc ? logs.acc.toFixed(4) : 'N/A'}, val_loss: ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'}`;
      }
    }]
  });

  $('training-status').textContent += ' — Training completed.';

  // Get validation preds
  const valPredsTensor = model.predict(valX);
  validationPredsArr = valPredsTensor.arraySync().map(x => x[0]);
  const valLabelsArr = valY.arraySync().map(x => x[0]);

  // store labels for metric computation
  validation.labels = valLabelsArr;
  validation.predictions = validationPredsArr;

  // Compute ROC and initial metrics
  computeROCAndPlot(validation.labels, validation.predictions);
  setDisabled('threshold-slider', false);
  setDisabled('predict-btn', false);
  setDisabled('export-btn', true); // only enable after predictions
  setDisabled('save-model-btn', false);

  // Dispose train tensors to free memory
  trainX.dispose(); trainY.dispose();
}

// --------------------
// ROC and metrics
// --------------------
function computeROCAndPlot(yTrue, yProb) {
  if (!yTrue || !yProb) return;

  const thresholds = Array.from({length: 101}, (_,i) => i / 100);
  const rocPoints = thresholds.map(t => {
    let tp=0, tn=0, fp=0, fn=0;
    yTrue.forEach((yt, i) => {
      const yp = (yProb[i] >= t) ? 1 : 0;
      if (yt === 1 && yp === 1) tp++;
      if (yt === 0 && yp === 1) fp++;
      if (yt === 0 && yp === 0) tn++;
      if (yt === 1 && yp === 0) fn++;
    });
    const tpr = tp / (tp + fn + 1e-9);
    const fpr = fp / (fp + tn + 1e-9);
    return { x: fpr, y: tpr };
  });

  // AUC approximate trapezoid:
  const sorted = rocPoints.slice().sort((a,b)=>a.x-b.x);
  let auc = 0;
  for (let i=1;i<sorted.length;i++){
    const x1 = sorted[i-1].x, x2 = sorted[i].x;
    const y1 = sorted[i-1].y, y2 = sorted[i].y;
    auc += (x2 - x1) * (y1 + y2) / 2;
  }

  tfvis.render.linechart({ name: 'ROC Curve (val)', tab: 'Evaluation' }, { values: sorted.map(p => ({ x: p.x, y: p.y })) }, { xLabel: 'FPR', yLabel: 'TPR' });

  // Add AUC to metrics area
  $('performance-metrics').innerHTML = `<p>AUC: ${auc.toFixed(4)}</p>`;
  updateMetrics(); // update confusion/precision/recall at current slider
}

// Update confusion matrix & precision/recall/F1 using threshold slider
function updateMetrics() {
  if (!validation || !validation.predictions || !validation.labels) return;
  const t = parseFloat($('threshold-slider').value);

  let tp=0, tn=0, fp=0, fn=0;
  validation.labels.forEach((yt, i) => {
    const yp = (validation.predictions[i] >= t) ? 1 : 0;
    if (yt === 1 && yp === 1) tp++;
    if (yt === 0 && yp === 0) tn++;
    if (yt === 0 && yp === 1) fp++;
    if (yt === 1 && yp === 0) fn++;
  });

  // Build confusion matrix
  $('confusion-matrix').innerHTML = `
    <table>
      <tr><th></th><th>Pred=0</th><th>Pred=1</th></tr>
      <tr><th>True=0</th><td>${tn}</td><td>${fp}</td></tr>
      <tr><th>True=1</th><td>${fn}</td><td>${tp}</td></tr>
    </table>
  `;

  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? 2 * (precision * recall) / (precision + recall) : 0;
  const accuracy = (tp + tn) / Math.max(1, (tp + tn + fp + fn));

  $('performance-metrics').innerHTML = `
    <p>AUC: ${($('performance-metrics').textContent.match(/AUC:/) ? $('performance-metrics').textContent.split('AUC:')[1].trim() : '')}</p>
    <p>Accuracy: ${(accuracy*100).toFixed(2)}%</p>
    <p>Precision: ${precision.toFixed(4)}</p>
    <p>Recall: ${recall.toFixed(4)}</p>
    <p>F1 Score: ${f1.toFixed(4)}</p>
  `;
}

// --------------------
// Predict on test set
// --------------------
async function predictTest() {
  if (!model || !preprocessedTest) return alert('Need model and preprocessed test data');

  $('prediction-output').textContent = 'Predicting...';

  const testX = tf.tensor2d(preprocessedTest.featuresArr);
  const predsTensor = model.predict(testX);
  testPredsArr = predsTensor.arraySync().map(x => x[0]);

  // Show first 12 probabilities
  $('prediction-output').innerHTML = `<p>Probabilities (first 12): ${testPredsArr.slice(0,12).map(p => p.toFixed(4)).join(', ')} …</p>`;
  setDisabled('export-btn', false);

  // cleanup
  testX.dispose();
}

// --------------------
// Export submission & probabilities & save model trigger
// --------------------
function exportResults() {
  if (!testPredsArr || !preprocessedTest) return alert('No predictions available - run Predict first');

  let submission = 'PassengerId,Survived\n';
  let probs = 'PassengerId,Probability\n';
  for (let i = 0; i < testPredsArr.length; i++) {
    const id = preprocessedTest.passengerIds[i] ?? i+1;
    const survived = testPredsArr[i] >= 0.5 ? 1 : 0;
    submission += `${id},${survived}\n`;
    probs += `${id},${testPredsArr[i].toFixed(6)}\n`;
  }
  downloadBlob(submission, 'submission.csv');
  downloadBlob(probs, 'probabilities.csv');
  $('export-status').textContent = 'Downloaded submission.csv and probabilities.csv.';
}

// small blob helper
function downloadBlob(text, filename) {
  const blob = new Blob([text], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// --------------------
// Utilities
// --------------------
function arrayColumn(arr, col) { return arr.map(r => r[col]); }
