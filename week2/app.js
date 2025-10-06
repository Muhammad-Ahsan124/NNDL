// =====================
// Titanic TF.js app.js
// =====================
//
// This file contains client-side logic to:
// - Load CSV files (robust to quoted commas)
// - Inspect & visualize basic stats (tfjs-vis)
// - Preprocess (impute, standardize, one-hot, optional family features)
// - Create a shallow neural classifier (Dense(16)->Dense(1, sigmoid))
// - Train with stratified 80/20 split, early stopping, and tfjs-vis live plots
// - Evaluate: ROC/AUC + threshold slider updates confusion matrix & metrics
// - Predict on test file and export submission/probabilities and model
//
// Schema can be changed at the top of this file (see constants).
//
// NOTE: This is purely browser-side and intended to run on GitHub Pages.
//
// ---------------------

// Global state
let trainData = null;
let testData = null;
let preprocessedTrain = null; // {featuresArr, labelsArr, featuresTensor, labelsTensor}
let preprocessedTest = null;  // {featuresArr, passengerIds}
let model = null;
let validationData = null; // {featuresTensor, labelsTensor}
let validationPredictions = null; // tensor
let testPredictions = null; // tensor

// ---------------------
// Schema (change here to reuse with another dataset)
// ---------------------
const TARGET_FEATURE = 'Survived';     // Binary target column name
const ID_FEATURE = 'PassengerId';      // ID column name
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];  // numeric columns to use
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];   // categorical columns to use
// ---------------------

// Utility: get element quickly
const $ = id => document.getElementById(id);

// ---------------------
// CSV loading & parsing
// ---------------------

// Load files provided in file inputs
async function loadData() {
    const trainFile = $('train-file').files[0];
    const testFile = $('test-file').files[0];
    const statusDiv = $('data-status');
    statusDiv.innerHTML = '';

    if (!trainFile || !testFile) {
        alert('Please upload both training (train.csv) and test (test.csv) files.');
        return;
    }

    statusDiv.innerHTML = 'Reading files...';

    try {
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);

        // Use robust CSV parser that handles quoted commas and double quotes
        trainData = parseCSV(trainText);
        testData = parseCSV(testText);

        statusDiv.innerHTML = `Loaded: train=${trainData.length} rows, test=${testData.length} rows`;

        // Enable inspect & preprocess buttons
        $('inspect-btn').disabled = false;
        $('preprocess-btn').disabled = false;
    } catch (err) {
        console.error(err);
        statusDiv.innerHTML = `Error loading files: ${err.message}`;
        alert('Error reading files. Check console for details.');
    }
}

// Read file as text (FileReader)
function readFile(file) {
    return new Promise((resolve, reject) => {
        const fr = new FileReader();
        fr.onload = e => resolve(e.target.result);
        fr.onerror = e => reject(new Error('Failed to read file'));
        fr.readAsText(file);
    });
}

// Robust CSV parser that handles quoted fields (commas inside quotes) and escaped double quotes ("")
function parseCSV(csvText) {
    // Normalize newlines and trim
    const text = csvText.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const lines = text.split('\n').filter(line => line.trim() !== '');

    if (lines.length === 0) return [];

    // Parse header (handles quoted headers)
    const parseLine = (line) => {
        const values = [];
        let current = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            // Handle escaped quotes ("")
            if (ch === '"' && line[i + 1] === '"') {
                current += '"';
                i++;
            } else if (ch === '"') {
                inQuotes = !inQuotes;
            } else if (ch === ',' && !inQuotes) {
                values.push(current);
                current = '';
            } else {
                current += ch;
            }
        }
        values.push(current);
        // Trim surrounding quotes and whitespace
        return values.map(v => {
            let r = v.trim();
            if (r.startsWith('"') && r.endsWith('"')) {
                r = r.slice(1, -1);
            }
            return r;
        });
    };

    const headers = parseLine(lines[0]);

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const rowVals = parseLine(lines[i]);
        // Some rows might have fewer columns (malformed) - pad with nulls
        while (rowVals.length < headers.length) rowVals.push('');
        const obj = {};
        headers.forEach((h, idx) => {
            let value = rowVals[idx] === '' ? null : rowVals[idx];
            // Convert numeric-looking values to numbers
            if (value !== null && !isNaN(value) && value !== '') {
                // But we also want to keep things like '001' as numbers are fine for features
                value = parseFloat(value);
            }
            obj[h] = value;
        });
        rows.push(obj);
    }

    return rows;
}

// ---------------------
// Inspect data & visualizations
// ---------------------
function inspectData() {
    if (!trainData || !Array.isArray(trainData) || trainData.length === 0) {
        alert('No training data loaded. Please upload train.csv first.');
        return;
    }

    // Preview first 10 rows
    const previewDiv = $('data-preview');
    previewDiv.innerHTML = '<h3>Preview (first 10 rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));

    // Stats: shape, survival rate, missing %
    const statsDiv = $('data-stats');
    const numCols = Object.keys(trainData[0]).length;
    const shapeText = `Shape: ${trainData.length} rows x ${numCols} columns`;

    const survivalCount = trainData.filter(r => r[TARGET_FEATURE] === 1).length;
    const survivalRate = ((survivalCount / trainData.length) * 100).toFixed(2);

    let missingHtml = '<h4>Missing (%)</h4><ul>';
    Object.keys(trainData[0]).forEach(col => {
        const missingCount = trainData.filter(r => r[col] === null || r[col] === undefined).length;
        const pct = ((missingCount / trainData.length) * 100).toFixed(2);
        missingHtml += `<li>${col}: ${pct}%</li>`;
    });
    missingHtml += '</ul>';

    statsDiv.innerHTML = `<p>${shapeText}</p><p>Survival: ${survivalCount}/${trainData.length} (${survivalRate}%)</p>${missingHtml}`;

    // Visualizations via tfjs-vis
    createVisualizations();

    // Enable preprocess (already enabled at load, but keep consistent)
    $('preprocess-btn').disabled = false;
}

function createPreviewTable(rows) {
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    Object.keys(rows[0]).forEach(k => {
        const th = document.createElement('th');
        th.textContent = k;
        trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach(r => {
        const tr = document.createElement('tr');
        Object.values(r).forEach(v => {
            const td = document.createElement('td');
            td.textContent = v === null || v === undefined ? 'NULL' : v;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

function createVisualizations() {
    const chartsDiv = $('charts');
    chartsDiv.innerHTML = '<h4>Charts (open tfjs-vis visor)</h4>';

    // Survival rate by Sex
    const groupSex = {};
    trainData.forEach(r => {
        const sex = r.Sex ?? 'unknown';
        if (!(sex in groupSex)) groupSex[sex] = { survived: 0, total: 0 };
        groupSex[sex].total++;
        if (r[TARGET_FEATURE] === 1) groupSex[sex].survived++;
    });
    const sexData = Object.entries(groupSex).map(([k, v]) => ({ x: k, y: (v.survived / v.total) * 100 }));
    tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Charts' }, sexData, { xLabel: 'Sex', yLabel: 'Survival %' });

    // Survival by Pclass
    const groupClass = {};
    trainData.forEach(r => {
        const cls = r.Pclass ?? 'unknown';
        if (!(cls in groupClass)) groupClass[cls] = { survived: 0, total: 0 };
        groupClass[cls].total++;
        if (r[TARGET_FEATURE] === 1) groupClass[cls].survived++;
    });
    const classData = Object.entries(groupClass).map(([k, v]) => ({ x: `Class ${k}`, y: (v.survived / v.total) * 100 }));
    tfvis.render.barchart({ name: 'Survival Rate by Class', tab: 'Charts' }, classData, { xLabel: 'Class', yLabel: 'Survival %' });
}

// ---------------------
// Preprocessing
// ---------------------
function preprocessData() {
    if (!trainData || !testData) {
        alert('Load train & test files before preprocessing.');
        return;
    }
    const output = $('preprocessing-output');
    output.innerHTML = 'Preprocessing...';

    try {
        // Compute imputation & standardization parameters from training data
        const ages = trainData.map(r => r.Age).filter(v => v !== null && v !== undefined && !isNaN(v));
        const fares = trainData.map(r => r.Fare).filter(v => v !== null && v !== undefined && !isNaN(v));
        const embarkedVals = trainData.map(r => r.Embarked).filter(v => v !== null && v !== undefined);

        const ageMedian = calculateMedian(ages);
        const fareMedian = calculateMedian(fares);
        const ageStd = calculateStdDev(ages) || 1;
        const fareStd = calculateStdDev(fares) || 1;
        const embarkedMode = calculateMode(embarkedVals);

        // Helper: extract features as array for a row
        const extractRowFeatures = (row) => {
            // Impute
            const age = row.Age !== null && row.Age !== undefined && !isNaN(row.Age) ? row.Age : ageMedian;
            const fare = row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare) ? row.Fare : fareMedian;
            const embarked = row.Embarked !== null && row.Embarked !== undefined ? row.Embarked : embarkedMode;

            // Standardize numeric features (use medians/std from train)
            const stdAge = (age - ageMedian) / ageStd;
            const stdFare = (fare - fareMedian) / fareStd;
            const sibsp = (row.SibSp !== null && row.SibSp !== undefined && !isNaN(row.SibSp)) ? row.SibSp : 0;
            const parch = (row.Parch !== null && row.Parch !== undefined && !isNaN(row.Parch)) ? row.Parch : 0;

            // One-hot encodings
            const pclassOH = oneHotEncode(row.Pclass, [1, 2, 3]); // Pclass usually 1/2/3
            const sexOH = oneHotEncode(row.Sex, ['male', 'female']);
            const embarkedOH = oneHotEncode(embarked, ['C', 'Q', 'S']);

            // base features
            let features = [stdAge, stdFare, sibsp, parch];
            features = features.concat(pclassOH, sexOH, embarkedOH);

            // optional family features
            if ($('add-family-features').checked) {
                const familySize = (sibsp || 0) + (parch || 0) + 1;
                const isAlone = familySize === 1 ? 1 : 0;
                features.push(familySize, isAlone);
            }
            return features;
        };

        // Build train arrays
        const trainFeaturesArr = [];
        const trainLabelsArr = [];
        trainData.forEach(r => {
            trainFeaturesArr.push(extractRowFeatures(r));
            // Ensure label is 0 or 1, fallback to 0 if missing
            trainLabelsArr.push((r[TARGET_FEATURE] === 1) ? 1 : 0);
        });

        // Build test arrays (and keep PassengerId)
        const testFeaturesArr = [];
        const testPassengerIds = [];
        testData.forEach(r => {
            testFeaturesArr.push(extractRowFeatures(r));
            testPassengerIds.push(r[ID_FEATURE]);
        });

        // Save preprocessed arrays (tensors will be made after stratified split)
        preprocessedTrain = {
            featuresArr: trainFeaturesArr,
            labelsArr: trainLabelsArr
        };
        preprocessedTest = {
            featuresArr: testFeaturesArr,
            passengerIds: testPassengerIds
        };

        // Show shapes
        output.innerHTML = `
            <p>Preprocessing finished.</p>
            <p>Training: ${trainFeaturesArr.length} samples, feature size=${trainFeaturesArr[0].length}</p>
            <p>Test: ${testFeaturesArr.length} samples, feature size=${testFeaturesArr[0].length}</p>
        `;

        // Enable model creation
        $('create-model-btn').disabled = false;
    } catch (err) {
        console.error(err);
        output.innerHTML = `Preprocessing error: ${err.message}`;
    }
}

// ---------------------
// Utilities for stats & encoding
// ---------------------
function calculateMedian(arr) {
    if (!arr || arr.length === 0) return 0;
    const sorted = arr.slice().sort((a,b) => a-b);
    const half = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) return (sorted[half - 1] + sorted[half]) / 2;
    return sorted[half];
}
function calculateMode(arr) {
    if (!arr || arr.length === 0) return null;
    const freq = {};
    let max = 0, mode = null;
    arr.forEach(v => {
        freq[v] = (freq[v] || 0) + 1;
        if (freq[v] > max) { max = freq[v]; mode = v; }
    });
    return mode;
}
function calculateStdDev(arr) {
    if (!arr || arr.length === 0) return 0;
    const mean = arr.reduce((s,x) => s + x, 0) / arr.length;
    const varr = arr.reduce((s,x) => s + Math.pow(x - mean, 2), 0) / arr.length;
    return Math.sqrt(varr);
}
function oneHotEncode(value, categories) {
    const out = new Array(categories.length).fill(0);
    const idx = categories.indexOf(value);
    if (idx !== -1) out[idx] = 1;
    return out;
}

// ---------------------
// Model creation
// ---------------------
function createModel() {
    if (!preprocessedTrain) {
        alert('Preprocess data first.');
        return;
    }

    const inputDim = preprocessedTrain.featuresArr[0].length;

    model = tf.sequential();
    // Single hidden layer with 16 units
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Display a simple summary
    const summaryDiv = $('model-summary');
    let summaryHTML = `<p>Model: Dense(16, relu) -> Dense(1, sigmoid)</p>`;
    summaryHTML += `<p>Input shape: [${inputDim}]</p>`;
    summaryHTML += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML = summaryHTML;

    $('train-btn').disabled = false;
}

// ---------------------
// Stratified split helper (works on arrays)
// ---------------------
function stratifiedSplitArrays(featuresArr, labelsArr, valRatio=0.2, seed=42) {
    // Group indices by label
    const groups = {};
    labelsArr.forEach((lbl, i) => {
        const key = lbl.toString();
        if (!groups[key]) groups[key] = [];
        groups[key].push(i);
    });

    // Shuffle helper (Fisher-Yates)
    function shuffle(array, seedVal) {
        let m = array.length, t, i;
        let random = mulberry32(seedVal || Date.now());
        while (m) {
            i = Math.floor(random() * m--);
            t = array[m];
            array[m] = array[i];
            array[i] = t;
        }
        return array;
    }
    function mulberry32(a) {
        return function() {
            var t = a += 0x6D2B79F5;
            t = Math.imul(t ^ t >>> 15, t | 1);
            t ^= t + Math.imul(t ^ t >>> 7, t | 61);
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        }
    }

    const trainIdx = [], valIdx = [];
    Object.values(groups).forEach(indices => {
        const shuffled = shuffle(indices.slice(), seed);
        const cut = Math.floor(shuffled.length * (1 - valRatio));
        trainIdx.push(...shuffled.slice(0, cut));
        valIdx.push(...shuffled.slice(cut));
    });

    // Build arrays
    const trainFeatures = trainIdx.map(i => featuresArr[i]);
    const trainLabels = trainIdx.map(i => labelsArr[i]);
    const valFeatures = valIdx.map(i => featuresArr[i]);
    const valLabels = valIdx.map(i => labelsArr[i]);

    return {
        trainFeatures, trainLabels, valFeatures, valLabels
    };
}

// ---------------------
// Training
// ---------------------
async function trainModel() {
    if (!model || !preprocessedTrain) {
        alert('Create model after preprocessing first.');
        return;
    }

    $('training-status').innerHTML = 'Preparing training...';

    try {
        // Perform stratified split on arrays
        const { trainFeatures, trainLabels, valFeatures, valLabels } = stratifiedSplitArrays(preprocessedTrain.featuresArr, preprocessedTrain.labelsArr, 0.2, 1234);

        // Convert to tensors
        const trainX = tf.tensor2d(trainFeatures);
        const trainY = tf.tensor1d(trainLabels).reshape([trainLabels.length, 1]);
        const valX = tf.tensor2d(valFeatures);
        const valY = tf.tensor1d(valLabels).reshape([valLabels.length, 1]);

        // Save references for later
        preprocessedTrain.featuresTensor = trainX;
        preprocessedTrain.labelsTensor = trainY;
        validationData = { featuresTensor: valX, labelsTensor: valY };

        // Callbacks: combine tfjs-vis callbacks and EarlyStopping + custom onEpochEnd status update
        const fitVisCallbacks = tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        );

        const earlyStopping = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 });

        // Train
        $('training-status').innerHTML = 'Training started...';
        await model.fit(trainX, trainY, {
            epochs: 50,
            batchSize: 32,
            validationData: [valX, valY],
            callbacks: [fitVisCallbacks, earlyStopping, {
                onEpochEnd: async (epoch, logs) => {
                    $('training-status').innerHTML = `Epoch ${epoch + 1} / 50 — loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
                }
            }]
        });

        $('training-status').innerHTML += '<br/><strong>Training finished.</strong>';

        // Make predictions on validation set
        validationPredictions = model.predict(validationData.featuresTensor);

        // Enable threshold slider, predict & export
        $('threshold-slider').disabled = false;
        // Remove previous listener if present, then add
        $('threshold-slider').oninput = updateMetrics;
        $('predict-btn').disabled = false;

        // Initial metrics calculation
        updateMetrics();
    } catch (err) {
        console.error(err);
        $('training-status').innerHTML = `Training error: ${err.message}`;
    }
}

// ---------------------
// Metrics & ROC/AUC
// ---------------------
async function updateMetrics() {
    if (!validationPredictions || !validationData) return;

    const threshold = parseFloat($('threshold-slider').value);
    $('threshold-value').textContent = threshold.toFixed(2);

    const predVals = validationPredictions.arraySync().map(v => v[0]); // flatten
    const trueVals = validationData.labelsTensor.arraySync().map(v => v[0]);

    // Confusion counts
    let tp=0, tn=0, fp=0, fn=0;
    for (let i = 0; i < predVals.length; i++) {
        const p = (predVals[i] >= threshold) ? 1 : 0;
        const t = trueVals[i];
        if (p === 1 && t === 1) tp++;
        if (p === 0 && t === 0) tn++;
        if (p === 1 && t === 0) fp++;
        if (p === 0 && t === 1) fn++;
    }

    // Update confusion matrix table
    const cmDiv = $('confusion-matrix');
    cmDiv.innerHTML = `
        <table>
            <tr><th></th><th>Pred Positive</th><th>Pred Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;

    // Metrics (guard against divide-by-zero)
    const precision = (tp + fp) ? tp / (tp + fp) : 0;
    const recall = (tp + fn) ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) ? 2 * (precision * recall) / (precision + recall) : 0;
    const accuracy = (tp + tn + fp + fn) ? (tp + tn) / (tp + tn + fp + fn) : 0;

    const perfDiv = $('performance-metrics');
    perfDiv.innerHTML = `
        <p>Accuracy: ${(accuracy*100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;

    // Plot ROC & compute AUC
    await plotROC(trueVals, predVals);
}

// Plot ROC curve (approx AUC via trapezoidal rule)
async function plotROC(trueLabels, predScores) {
    // Build ROC points
    const thresholds = Array.from({length: 101}, (_,i) => i/100);
    const rocPoints = thresholds.map(th => {
        let tp=0, tn=0, fp=0, fn=0;
        for (let i=0;i<predScores.length;i++) {
            const pred = predScores[i] >= th ? 1 : 0;
            const t = trueLabels[i];
            if (t === 1) { if (pred === 1) tp++; else fn++; }
            else { if (pred === 1) fp++; else tn++; }
        }
        const tpr = (tp + fn) ? tp / (tp + fn) : 0;
        const fpr = (fp + tn) ? fp / (fp + tn) : 0;
        return { fpr, tpr };
    });

    // Approximate AUC (sort by fpr asc)
    const sorted = rocPoints.sort((a,b) => a.fpr - b.fpr);
    let auc = 0;
    for (let i=1;i<sorted.length;i++) {
        const x1 = sorted[i-1].fpr, x2 = sorted[i].fpr;
        const y1 = sorted[i-1].tpr, y2 = sorted[i].tpr;
        auc += (x2 - x1) * (y1 + y2) / 2;
    }

    // Render ROC in tfjs-vis
    const lineData = sorted.map(pt => ({ x: pt.fpr, y: pt.tpr }));
    tfvis.render.linechart({ name: 'ROC Curve', tab: 'Evaluation' }, { values: lineData }, { xLabel: 'FPR', yLabel: 'TPR', width: 400, height: 400 });

    // Append AUC to performance metrics
    const perfDiv = $('performance-metrics');
    perfDiv.innerHTML += `<p>AUC: ${auc.toFixed(4)}</p>`;
}

// ---------------------
// Prediction & Export
// ---------------------
async function predict() {
    if (!model || !preprocessedTest) {
        alert('Model or test data missing. Train and preprocess first.');
        return;
    }

    const out = $('prediction-output');
    out.innerHTML = 'Predicting...';

    try {
        const testX = tf.tensor2d(preprocessedTest.featuresArr);
        testPredictions = model.predict(testX);
        const predVals = testPredictions.arraySync().map(v => v[0]);

        // Build results array
        const results = preprocessedTest.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predVals[i] >= 0.5 ? 1 : 0,
            Probability: predVals[i]
        }));

        out.innerHTML = '<h4>Predictions (first 10)</h4>';
        out.appendChild(createPredictionTable(results.slice(0, 10)));
        out.innerHTML += `<p>Total predictions: ${results.length}</p>`;

        $('export-btn').disabled = false;
    } catch (err) {
        console.error(err);
        out.innerHTML = `Prediction error: ${err.message}`;
    }
}

function createPredictionTable(rows) {
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        trh.appendChild(th);
    });
    thead.appendChild(trh); table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach(r => {
        const tr = document.createElement('tr');
        const tdId = document.createElement('td'); tdId.textContent = r.PassengerId; tr.appendChild(tdId);
        const tdS = document.createElement('td'); tdS.textContent = r.Survived; tr.appendChild(tdS);
        const tdP = document.createElement('td'); tdP.textContent = r.Probability.toFixed(4); tr.appendChild(tdP);
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

// Export submission & probabilities & save model (downloads)
async function exportResults() {
    if (!testPredictions || !preprocessedTest) {
        alert('No predictions to export. Run Predict first.');
        return;
    }

    const status = $('export-status');
    status.innerHTML = 'Preparing export...';

    try {
        const predVals = testPredictions.arraySync().map(v => v[0]);

        // submission.csv (PassengerId,Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTest.passengerIds.forEach((id, i) => {
            submissionCSV += `${id},${predVals[i] >= 0.5 ? 1 : 0}\n`;
        });

        // probabilities.csv (PassengerId,Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTest.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${predVals[i].toFixed(6)}\n`;
        });

        // Trigger downloads
        const blobSub = new Blob([submissionCSV], { type: 'text/csv' });
        const linkSub = document.createElement('a');
        linkSub.href = URL.createObjectURL(blobSub);
        linkSub.download = 'submission.csv';
        document.body.appendChild(linkSub);
        linkSub.click();
        linkSub.remove();

        const blobProb = new Blob([probabilitiesCSV], { type: 'text/csv' });
        const linkProb = document.createElement('a');
        linkProb.href = URL.createObjectURL(blobProb);
        linkProb.download = 'probabilities.csv';
        document.body.appendChild(linkProb);
        linkProb.click();
