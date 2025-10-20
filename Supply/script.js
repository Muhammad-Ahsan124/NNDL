// Global variables
let dataset = null;
let model = null;
let scaler = null;
let labelEncoders = {};
let trainingHistory = { loss: [], val_loss: [], accuracy: [], val_accuracy: [] };
let trainingChart = null;

// File upload handling
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileInfo').style.display = 'block';
        
        // Preview file info
        Papa.parse(file, {
            header: true,
            preview: 1,
            complete: function(results) {
                document.getElementById('rowCount').textContent = 'Preview available';
            }
        });
    }
});

// Load dataset function
async function loadDataset() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file first');
        return;
    }

    showLoading('Loading dataset...');
    
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: function(results) {
            dataset = results.data.filter(row => 
                Object.values(row).some(val => val !== null && val !== undefined && val !== '')
            );
            
            hideLoading();
            
            if (dataset.length === 0) {
                alert('No valid data found in the file');
                return;
            }

            console.log('Dataset loaded:', dataset.length, 'rows');
            
            // Show next sections
            document.getElementById('edaSection').style.display = 'block';
            document.getElementById('trainingSection').style.display = 'block';
            document.getElementById('predictionSection').style.display = 'block';
            document.getElementById('evaluationSection').style.display = 'block';
            
            // Populate prediction dropdowns
            populatePredictionFields();
            
            alert(`Dataset loaded successfully! ${dataset.length} rows processed.`);
        },
        error: function(error) {
            hideLoading();
            alert('Error loading file: ' + error);
        }
    });
}

// Populate prediction form fields
function populatePredictionFields() {
    if (!dataset || dataset.length === 0) return;
    
    // Get unique asset IDs
    const assetIds = [...new Set(dataset.map(row => row.Asset_ID))];
    const assetSelect = document.getElementById('predAssetId');
    assetSelect.innerHTML = '';
    assetIds.forEach(id => {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = id;
        assetSelect.appendChild(option);
    });
    
    // Set default values based on first row
    const sampleRow = dataset[0];
    document.getElementById('predLatitude').value = sampleRow.Latitude || 0;
    document.getElementById('predLongitude').value = sampleRow.Longitude || 0;
    document.getElementById('predInventory').value = sampleRow.Inventory_Level || 0;
    document.getElementById('predTemperature').value = sampleRow.Temperature || 0;
    document.getElementById('predHumidity').value = sampleRow.Humidity || 0;
    document.getElementById('predWaitingTime').value = sampleRow.Waiting_Time || 0;
    document.getElementById('predTransactionAmount').value = sampleRow.User_Transaction_Amount || 0;
    document.getElementById('predPurchaseFrequency').value = sampleRow.User_Purchase_Frequency || 0;
    document.getElementById('predAssetUtilization').value = sampleRow.Asset_Utilization || 0;
    document.getElementById('predDemandForecast').value = sampleRow.Demand_Forecast || 0;
}

// Perform EDA
function performEDA() {
    if (!dataset) {
        alert('Please load dataset first');
        return;
    }

    showLoading('Generating EDA plots...');
    
    // Simulate processing time
    setTimeout(() => {
        generateEDACharts();
        hideLoading();
    }, 1000);
}

// Generate EDA charts
function generateEDACharts() {
    // Chart 1: Delay Distribution
    const delayCounts = dataset.reduce((acc, row) => {
        const delay = row.Logistics_Delay;
        acc[delay] = (acc[delay] || 0) + 1;
        return acc;
    }, {});
    
    const ctx1 = document.getElementById('chart1').getContext('2d');
    new Chart(ctx1, {
        type: 'pie',
        data: {
            labels: ['No Delay (0)', 'Delay (1)'],
            datasets: [{
                data: [delayCounts[0] || 0, delayCounts[1] || 0],
                backgroundColor: ['#4CAF50', '#F44336']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Delay Distribution'
                }
            }
        }
    });

    // Chart 2: Shipment Status Distribution
    const statusCounts = dataset.reduce((acc, row) => {
        const status = row.Shipment_Status;
        acc[status] = (acc[status] || 0) + 1;
        return acc;
    }, {});
    
    const ctx2 = document.getElementById('chart2').getContext('2d');
    new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: Object.keys(statusCounts),
            datasets: [{
                label: 'Count',
                data: Object.values(statusCounts),
                backgroundColor: '#667eea'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Shipment Status Distribution'
                }
            }
        }
    });

    // Chart 3: Temperature vs Delay
    const tempRanges = ['<20°C', '20-25°C', '25-30°C', '>30°C'];
    const tempDelayData = tempRanges.map(range => {
        let filteredData = [];
        if (range === '<20°C') filteredData = dataset.filter(row => row.Temperature < 20);
        else if (range === '20-25°C') filteredData = dataset.filter(row => row.Temperature >= 20 && row.Temperature < 25);
        else if (range === '25-30°C') filteredData = dataset.filter(row => row.Temperature >= 25 && row.Temperature < 30);
        else filteredData = dataset.filter(row => row.Temperature >= 30);
        
        const delays = filteredData.filter(row => row.Logistics_Delay === 1).length;
        return delays;
    });
    
    const ctx3 = document.getElementById('chart3').getContext('2d');
    new Chart(ctx3, {
        type: 'line',
        data: {
            labels: tempRanges,
            datasets: [{
                label: 'Number of Delays',
                data: tempDelayData,
                borderColor: '#FF6B6B',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Temperature vs Delays'
                }
            }
        }
    });
}

// Train model function
async function trainModel() {
    if (!dataset) {
        alert('Please load dataset first');
        return;
    }

    const modelType = document.getElementById('modelType').value;
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);

    showLoading('Preparing data for training...');
    
    // Prepare data
    const { features, labels, featureNames } = prepareTrainingData();
    
    // Split data
    const splitIndex = Math.floor(features.length * 0.8);
    const trainFeatures = features.slice(0, splitIndex);
    const trainLabels = labels.slice(0, splitIndex);
    const testFeatures = features.slice(splitIndex);
    const testLabels = labels.slice(splitIndex);
    
    // Convert to tensors
    const xTrain = tf.tensor2d(trainFeatures);
    const yTrain = tf.tensor1d(trainLabels);
    const xTest = tf.tensor2d(testFeatures);
    const yTest = tf.tensor1d(testLabels);
    
    // Build model
    model = buildModel(modelType, featureNames.length);
    
    // Show training progress
    document.getElementById('trainingProgress').style.display = 'block';
    
    // Initialize training chart
    initializeTrainingChart();
    
    // Train model
    await trainModelWithProgress(model, xTrain, yTrain, xTest, yTest, epochs, batchSize);
    
    // Clean up tensors
    xTrain.dispose();
    yTrain.dispose();
    xTest.dispose();
    yTest.dispose();
    
    hideLoading();
    alert('Model training completed!');
}

// Prepare training data
function prepareTrainingData() {
    const numericFeatures = [];
    const labels = [];
    
    // Define feature columns (excluding timestamp and asset_id for simplicity)
    const featureColumns = [
        'Latitude', 'Longitude', 'Inventory_Level', 'Temperature', 'Humidity', 
        'Waiting_Time', 'User_Transaction_Amount', 'User_Purchase_Frequency', 
        'Asset_Utilization', 'Demand_Forecast'
    ];
    
    // Encode categorical variables
    const categoricalColumns = ['Shipment_Status', 'Traffic_Status', 'Logistics_Delay_Reason'];
    labelEncoders = {};
    
    categoricalColumns.forEach(col => {
        const uniqueValues = [...new Set(dataset.map(row => row[col]))];
        labelEncoders[col] = {};
        uniqueValues.forEach((val, idx) => {
            labelEncoders[col][val] = idx;
        });
    });
    
    // Prepare features and labels
    dataset.forEach(row => {
        if (row.Logistics_Delay !== undefined && row.Logistics_Delay !== null) {
            const features = [];
            
            // Add numeric features
            featureColumns.forEach(col => {
                features.push(parseFloat(row[col]) || 0);
            });
            
            // Add encoded categorical features
            categoricalColumns.forEach(col => {
                features.push(labelEncoders[col][row[col]] || 0);
            });
            
            numericFeatures.push(features);
            labels.push(parseInt(row.Logistics_Delay));
        }
    });
    
    // Scale features
    scaler = {
        mean: [],
        std: []
    };
    
    const featureCount = numericFeatures[0].length;
    for (let i = 0; i < featureCount; i++) {
        const column = numericFeatures.map(row => row[i]);
        scaler.mean[i] = tf.mean(column).dataSync()[0];
        scaler.std[i] = tf.moments(column).variance.dataSync()[0] ** 0.5;
        
        // Apply scaling
        for (let j = 0; j < numericFeatures.length; j++) {
            numericFeatures[j][i] = (numericFeatures[j][i] - scaler.mean[i]) / (scaler.std[i] || 1);
        }
    }
    
    return {
        features: numericFeatures,
        labels: labels,
        featureNames: [...featureColumns, ...categoricalColumns]
    };
}

// Build model
function buildModel(modelType, inputDim) {
    const model = tf.sequential();
    
    if (modelType === 'LSTM') {
        model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true,
            inputShape: [10, inputDim]
        }));
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.lstm({ units: 32 }));
    } else { // GRU
        model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            inputShape: [10, inputDim]
        }));
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.gru({ units: 32 }));
    }
    
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

// Initialize training chart
function initializeTrainingChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#FF6B6B',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#4ECDC4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Training Progress'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Train model with progress tracking
async function trainModelWithProgress(model, xTrain, yTrain, xTest, yTest, epochs, batchSize) {
    const batchCount = Math.ceil(xTrain.shape[0] / batchSize);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        let epochLoss = 0;
        let epochAcc = 0;
        
        for (let batch = 0; batch < batchCount; batch++) {
            const start = batch * batchSize;
            const end = Math.min(start + batchSize, xTrain.shape[0]);
            
            const xBatch = xTrain.slice([start, 0], [end - start, -1]);
            const yBatch = yTrain.slice([start], [end - start]);
            
            const history = await model.trainOnBatch(xBatch, yBatch);
            epochLoss += history[0];
            epochAcc += history[1];
            
            xBatch.dispose();
            yBatch.dispose();
        }
        
        // Calculate validation metrics
        const valHistory = await model.evaluate(xTest, yTest);
        const valLoss = valHistory[0].dataSync()[0];
        const valAcc = valHistory[1].dataSync()[0];
        
        valHistory[0].dispose();
        valHistory[1].dispose();
        
        // Update history
        trainingHistory.loss.push(epochLoss / batchCount);
        trainingHistory.accuracy.push(epochAcc / batchCount);
        trainingHistory.val_loss.push(valLoss);
        trainingHistory.val_accuracy.push(valAcc);
        
        // Update UI
        updateTrainingProgress(epoch + 1, epochs);
        updateTrainingChart();
        
        // Add small delay for smooth animation
        await tf.nextFrame();
    }
}

// Update training progress
function updateTrainingProgress(currentEpoch, totalEpochs) {
    const progress = (currentEpoch / totalEpochs) * 100;
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressText').textContent = `Epoch ${currentEpoch}/${totalEpochs}`;
}

// Update training chart
function updateTrainingChart() {
    if (!trainingChart) return;
    
    const epochs = Array.from({length: trainingHistory.loss.length}, (_, i) => i + 1);
    
    trainingChart.data.labels = epochs;
    trainingChart.data.datasets[0].data = trainingHistory.loss;
    trainingChart.data.datasets[1].data = trainingHistory.val_loss;
    trainingChart.update('none');
}

// Make prediction
async function makePrediction() {
    if (!model) {
        alert('Please train the model first');
        return;
    }

    // Get input values
    const inputFeatures = getPredictionInput();
    
    // Prepare features
    const features = preparePredictionFeatures(inputFeatures);
    
    // Make prediction
    const inputTensor = tf.tensor2d([features]);
    const prediction = await model.predict(inputTensor).data();
    const probability = prediction[0];
    
    // Display result
    displayPredictionResult(probability, inputFeatures);
    
    inputTensor.dispose();
}

// Get prediction input
function getPredictionInput() {
    return {
        Asset_ID: document.getElementById('predAssetId').value,
        Latitude: parseFloat(document.getElementById('predLatitude').value),
        Longitude: parseFloat(document.getElementById('predLongitude').value),
        Inventory_Level: parseFloat(document.getElementById('predInventory').value),
        Shipment_Status: document.getElementById('predShipmentStatus').value,
        Temperature: parseFloat(document.getElementById('predTemperature').value),
        Humidity: parseFloat(document.getElementById('predHumidity').value),
        Traffic_Status: document.getElementById('predTrafficStatus').value,
        Waiting_Time: parseFloat(document.getElementById('predWaitingTime').value),
        User_Transaction_Amount: parseFloat(document.getElementById('predTransactionAmount').value),
        User_Purchase_Frequency: parseFloat(document.getElementById('predPurchaseFrequency').value),
        Logistics_Delay_Reason: document.getElementById('predDelayReason').value,
        Asset_Utilization: parseFloat(document.getElementById('predAssetUtilization').value),
        Demand_Forecast: parseFloat(document.getElementById('predDemandForecast').value)
    };
}

// Prepare prediction features
function preparePredictionFeatures(input) {
    const features = [];
    
    // Numeric features
    const numericFeatures = [
        input.Latitude, input.Longitude, input.Inventory_Level, input.Temperature,
        input.Humidity, input.Waiting_Time, input.User_Transaction_Amount,
        input.User_Purchase_Frequency, input.Asset_Utilization, input.Demand_Forecast
    ];
    
    // Categorical features
    const categoricalFeatures = [
        labelEncoders.Shipment_Status[input.Shipment_Status] || 0,
        labelEncoders.Traffic_Status[input.Traffic_Status] || 0,
        labelEncoders.Logistics_Delay_Reason[input.Logistics_Delay_Reason] || 0
    ];
    
    // Combine and scale
    const allFeatures = [...numericFeatures, ...categoricalFeatures];
    
    return allFeatures.map((val, idx) => {
        return (val - scaler.mean[idx]) / (scaler.std[idx] || 1);
    });
}

// Display prediction result
function displayPredictionResult(probability, input) {
    const resultDiv = document.getElementById('predictionResult');
    const resultContent = document.getElementById('resultContent');
    
    const percentage = (probability * 100).toFixed(1);
    const isDelay = probability > 0.5;
    
    let html = `
        <div style="text-align: center; margin-bottom: 15px;">
            <h3 style="font-size: 1.5rem; margin-bottom: 10px;">
                ${isDelay ? '🚨 HIGH DELAY RISK' : '✅ LOW DELAY RISK'}
            </h3>
            <p style="font-size: 1.2rem; font-weight: bold;">
                Probability: ${percentage}%
            </p>
        </div>
    `;
    
    if (isDelay) {
        html += `
            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                <h4>🚨 Recommended Actions:</h4>
                <ul style="text-align: left; margin-top: 10px;">
                    <li>Consider alternative routes</li>
                    <li>Check vehicle maintenance status</li>
                    <li>Notify customer about potential delay</li>
                    <li>Allocate additional resources</li>
                </ul>
            </div>
        `;
    } else {
        html += `
            <div style="background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
                <h4>✅ Status: Normal Operations</h4>
                <p>Current logistics parameters indicate smooth operations.</p>
            </div>
        `;
    }
    
    resultContent.innerHTML = html;
    resultDiv.className = isDelay ? 'delay' : 'no-delay';
    resultDiv.style.display = 'block';
}

// Evaluate model
async function evaluateModel() {
    if (!model || !dataset) {
        alert('Please train the model first');
        return;
    }

    showLoading('Evaluating model...');
    
    // Prepare test data
    const { features, labels } = prepareTrainingData();
    const splitIndex = Math.floor(features.length * 0.8);
    const testFeatures = features.slice(splitIndex);
    const testLabels = labels.slice(splitIndex);
    
    // Make predictions
    const xTest = tf.tensor2d(testFeatures);
    const predictions = await model.predict(xTest).data();
    
    // Calculate metrics
    const { accuracy, precision, recall, f1, confusionMatrix } = calculateMetrics(
        testLabels, 
        Array.from(predictions).map(p => p > 0.5 ? 1 : 0)
    );
    
    // Display results
    displayEvaluationResults(accuracy, precision, recall, f1, confusionMatrix);
    
    xTest.dispose();
    hideLoading();
}

// Calculate evaluation metrics
function calculateMetrics(trueLabels, predLabels) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === 1 && predLabels[i] === 1) tp++;
        else if (trueLabels[i] === 0 && predLabels[i] === 1) fp++;
        else if (trueLabels[i] === 0 && predLabels[i] === 0) tn++;
        else if (trueLabels[i] === 1 && predLabels[i] === 0) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    const confusionMatrix = [[tn, fp], [fn, tp]];
    
    return { accuracy, precision, recall, f1, confusionMatrix };
}

// Display evaluation results
function displayEvaluationResults(accuracy, precision, recall, f1, confusionMatrix) {
    // Update metric cards
    document.getElementById('accuracy').textContent = `${(accuracy * 100).toFixed(1)}%`;
    document.getElementById('precision').textContent = `${(precision * 100).toFixed(1)}%`;
    document.getElementById('recall').textContent = `${(recall * 100).toFixed(1)}%`;
    document.getElementById('f1').textContent = `${(f1 * 100).toFixed(1)}%`;
    
    // Create confusion matrix chart
    const ctx = document.getElementById('confusionMatrix').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
            datasets: [{
                label: 'Count',
                data: [confusionMatrix[0][0], confusionMatrix[0][1], confusionMatrix[1][0], confusionMatrix[1][1]],
                backgroundColor: ['#4CAF50', '#FF9800', '#FF9800', '#4CAF50']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Confusion Matrix'
                }
            }
        }
    });
    
    document.getElementById('evaluationResults').style.display = 'block';
}

// Utility functions
function showLoading(message = 'Processing...') {
    // Simple loading indicator
    const loading = document.createElement('div');
    loading.id = 'loadingOverlay';
    loading.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 1.2rem;
        z-index: 1000;
    `;
    loading.innerHTML = `
        <div style="text-align: center;">
            <div class="spinner" style="border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 15px;"></div>
            <p>${message}</p>
        </div>
    `;
    document.body.appendChild(loading);
}

function hideLoading() {
    const loading = document.getElementById('loadingOverlay');
    if (loading) {
        loading.remove();
    }
}

// Add CSS for spinner animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);
