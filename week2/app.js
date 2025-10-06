// ========== Global State ==========
let trainData = null;
let testData = null;
let model = null;
let preprocessedTrain = null;
let preprocessedTest = null;
let validationData = null;
let validationPredictions = null;
let testPredictions = null;

// Schema (change for reuse)
const TARGET = 'Survived';
const ID = 'PassengerId';

// ========== Load Data ==========
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    const status = document.getElementById('data-status');
    if (!trainFile || !testFile) {
        alert('Please upload both train.csv and test.csv.');
        return;
    }
    status.textContent = 'Loading data...';
    try {
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);
        trainData = parseCSV(trainText);
        testData = parseCSV(testText);
        status.textContent = `Data loaded! Training: ${trainData.length} rows, Test: ${testData.length} rows.`;
        document.getElementById('inspect-btn').disabled = false;
        document.getElementById('preprocess-btn').disabled = false;
    } catch (err) {
        console.error(err);
        status.textContent = `Error: ${err.message}`;
    }
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = () => reject(new Error('File read error.'));
        reader.readAsText(file);
    });
}

// ✅ Robust CSV parser (handles commas, quotes, Windows newlines)
function parseCSV(csvText) {
    const text = csvText.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
    const lines = text.split('\n').filter(line => line.trim() !== '');
    if (lines.length < 2) throw new Error('CSV missing data rows.');

    function splitCSV(line) {
        const result = [];
        let curr = '', inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const c = line[i];
            if (c === '"' && line[i + 1] === '"') { curr += '"'; i++; }
            else if (c === '"') inQuotes = !inQuotes;
            else if (c === ',' && !inQuotes) { result.push(curr); curr = ''; }
            else curr += c;
        }
        result.push(curr);
        return result.map(v => v.trim().replace(/^"|"$/g, ''));
    }

    const headers = splitCSV(lines[0]);
    return lines.slice(1).map(line => {
        const values = splitCSV(line);
        const obj = {};
        headers.forEach((h, i) => {
            let v = values[i] ?? null;
            if (v === '') v = null;
            if (v !== null && !isNaN(v)) v = parseFloat(v);
            obj[h] = v;
        });
        return obj;
    });
}

// ========== Data Inspection ==========
function inspectData() {
    if (!trainData) return alert('Please load data first.');
    const preview = document.getElementById('data-preview');
    const stats = document.getElementById('data-stats');

    preview.innerHTML = '<h3>Preview (First 10 Rows)</h3>' + createTable(trainData.slice(0, 10));
    const missing = Object.keys(trainData[0]).map(col => {
        const miss = trainData.filter(r => r[col] == null).length;
        return `<li>${col}: ${(miss / trainData.length * 100).toFixed(1)}%</li>`;
    }).join('');
    stats.innerHTML = `<p>Total Rows: ${trainData.length}</p><ul>${missing}</ul>`;
}

function createTable(data) {
    const cols = Object.keys(data[0]);
    let html = '<table><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
    data.forEach(row => {
        html += '<tr>' + cols.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>';
    });
    return html + '</table>';
}

// ========== Preprocessing ==========
function preprocessData() {
    const output = document.getElementById('preprocessing-output');
    output.textContent = 'Preprocessing...';
    const median = arr => {
        const s = arr.filter(v => v != null).sort((a, b) => a - b);
        return s[Math.floor(s.length / 2)];
    };
    const std = arr => {
        const m = arr.reduce((a, b) => a + b, 0) / arr.length;
        return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
    };
    const ageMedian = median(trainData.map(r => r.Age));
    const fareMedian = median(trainData.map(r => r.Fare));
    const ageStd = std(trainData.map(r => r.Age || ageMedian)) || 1;
    const fareStd = std(trainData.map(r => r.Fare || fareMedian)) || 1;
    const embarkedMode = mode(trainData.map(r => r.Embarked));

    function mode(arr) {
        const freq = {}; arr.forEach(v => { if (v) freq[v] = (freq[v] || 0) + 1; });
        return Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'S';
    }

    function oneHot(value, cats) {
        return cats.map(c => (c === value ? 1 : 0));
    }

    function extract(row) {
        const age = (row.Age ?? ageMedian);
        const fare = (row.Fare ?? fareMedian);
        const ageNorm = (age - ageMedian) / ageStd;
        const fareNorm = (fare - fareMedian) / fareStd;
        const pclass = oneHot(row.Pclass, [1, 2, 3]);
        const sex = oneHot(row.Sex, ['male', 'female']);
        const embarked = oneHot(row.Embarked ?? embarkedMode, ['C', 'Q', 'S']);
        const sibsp = row.SibSp ?? 0;
        const parch = row.Parch ?? 0;
        let features = [ageNorm, fareNorm, sibsp, parch, ...pclass, ...sex, ...embarked];
        if (document.getElementById('add-family-features').checked) {
            const fam = sibsp + parch + 1;
            features.push(fam, fam === 1 ? 1 : 0);
        }
        return features;
    }

    const trainX = [], trainY = [];
    trainData.forEach(r => {
        trainX.push(extract(r));
        trainY.push(r[TARGET]);
    });
    const testX = [], ids = [];
    testData.forEach(r => {
        testX.push(extract(r));
        ids.push(r[ID]);
    });

    preprocessedTrain = { X: tf.tensor2d(trainX), y: tf.tensor1d(trainY) };
    preprocessedTest = { X: tf.tensor2d(testX), ids };

    output.innerHTML = `
        <p>Preprocessing complete.</p>
        <p>Train: ${trainX.length} samples × ${trainX[0].length} features</p>
        <p>Test: ${testX.length} samples</p>`;
    document.getElementById('create-model-btn').disabled = false;
}

// ========== Model Setup ==========
function createModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [preprocessedTrain.X.shape[1]] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    document.getElementById('model-summary').innerHTML =
        `<p>Model: Dense(16,relu) → Dense(1,sigmoid)</p><p>Params: ${model.countParams()}</p>`;
    document.getElementById('train-btn').disabled = false;
}

// ========== Training ==========
async function trainModel() {
    const status = document.getElementById('training-status');
    status.textContent = 'Training...';

    const total = preprocessedTrain.X.shape[0];
    const split = Math.floor(total * 0.8);
    const trainX = preprocessedTrain.X.slice([0, 0], [split, -1]);
    const trainY = preprocessedTrain.y.slice([0], [split]);
    const valX = preprocessedTrain.X.slice([split, 0], [-1, -1]);
    const valY = preprocessedTrain.y.slice([split], [-1]);
    validationData = { valX, valY };

    const fitVis = tfvis.show.fitCallbacks(
        { name: 'Training', tab: 'Training' },
        ['loss', 'val_loss', 'acc', 'val_acc'],
        { callbacks: ['onEpochEnd'] }
    );

    await model.fit(trainX, trainY, {
        epochs: 50,
        batchSize: 32,
        validationData: [valX, valY],
        callbacks: [fitVis, tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 }), {
            onEpochEnd: (e, logs) => {
                status.textContent = `Epoch ${e + 1}/50 - loss: ${logs.loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
            }
        }]
    });
    status.innerHTML += '<br>Training done.';
    validationPredictions = model.predict(valX);
    document.getElementById('threshold-slider').disabled = false;
    document.getElementById('threshold-slider').oninput = updateMetrics;
    updateMetrics();
    document.getElementById('predict-btn').disabled = false;
}

// ========== Metrics ==========
async function updateMetrics() {
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    const preds = (await validationPredictions.array()).flat();
    const actuals = (await validationData.valY.array()).flat();

    let tp = 0, tn = 0, fp = 0, fn = 0;
    preds.forEach((p, i) => {
        const a = actuals[i];
        const pred = p >= threshold ? 1 : 0;
        if (pred === 1 && a === 1) tp++;
        else if (pred === 0 && a === 0) tn++;
        else if (pred === 1 && a === 0) fp++;
        else fn++;
    });

    document.getElementById('confusion-matrix').innerHTML = `
        <table>
            <tr><th></th><th>Pred+</th><th>Pred-</th></tr>
            <tr><th>Actual+</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual-</th><td>${fp}</td><td>${tn}</td></tr>
        </table>`;

    const precision = tp / (tp + fp + 1e-6);
    const recall = tp / (tp + fn + 1e-6);
    const f1 = 2 * (precision * recall) / (precision + recall + 1e-6);
    const acc = (tp + tn) / (tp + tn + fp + fn);

    document.getElementById('performance-metrics').innerHTML =
        `<p>Accuracy: ${(acc * 100).toFixed(2)}%</p>
         <p>Precision: ${precision.toFixed(4)}</p>
         <p>Recall: ${recall
