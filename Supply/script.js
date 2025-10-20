// TensorFlow.js model and prediction logic
class LogisticsPredictor {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.init();
    }

    async init() {
        await this.loadModel();
        this.initializeCharts();
        this.setupEventListeners();
    }

    async loadModel() {
        try {
            // In a real scenario, you would load a pre-trained TensorFlow.js model
            // For demonstration, we'll create a simple neural network
            this.model = await this.createDemoModel();
            this.isModelLoaded = true;
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
            // Fallback to demo model
            this.model = await this.createDemoModel();
            this.isModelLoaded = true;
        }
    }

    async createDemoModel() {
        // Create a simple model for demonstration
        // In production, you would load a pre-trained LSTM model
        const model = tf.sequential({
            layers: [
                tf.layers.dense({inputShape: [10, 13], units: 32, activation: 'relu'}),
                tf.layers.dense({units: 16, activation: 'relu'}),
                tf.layers.dense({units: 1, activation: 'sigmoid'})
            ]
        });

        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    async predictDelayProbability(inputData) {
        if (!this.isModelLoaded) {
            throw new Error('Model not loaded');
        }

        // Prepare input data for LSTM
        const sequence = LogisticsData.prepareInput(inputData);
        
        // Convert to tensor
        const inputTensor = tf.tensor3d([sequence], [1, 10, 13]);
        
        // Make prediction
        const prediction = await this.model.predict(inputTensor);
        const probability = await prediction.data();
        
        // Clean up tensors
        inputTensor.dispose();
        prediction.dispose();
        
        return probability[0];
    }

    initializeCharts() {
        const monthlyData = PerformanceMetrics.getMonthlyData();
        
        const ctx = document.getElementById('performanceChart').getContext('2d');
        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: monthlyData.labels,
                datasets: [
                    {
                        label: 'Actual Delays',
                        data: monthlyData.delays,
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Predicted Delays',
                        data: monthlyData.predictions,
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Monthly Delay Reduction Trend',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Delays'
                        }
                    }
                }
            }
        });
    }

    setupEventListeners() {
        // Add real-time validation
        const inputs = document.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('change', this.validateInput.bind(this));
        });

        // Add keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                predictDelay();
            }
        });
    }

    validateInput(event) {
        const input = event.target;
        const value = parseFloat(input.value);
        
        if (input.type === 'number') {
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            
            if (value < min || value > max) {
                input.style.borderColor = '#ff6b6b';
                input.title = `Value must be between ${min} and ${max}`;
            } else {
                input.style.borderColor = '#e1e5e9';
                input.title = '';
            }
        }
    }

    displayResult(probability, confidence) {
        const resultDiv = document.getElementById('result');
        const predictionText = document.getElementById('predictionText');
        const confidenceDiv = document.getElementById('confidence');

        const isDelay = probability > 0.5;
        const percentage = (probability * 100).toFixed(1);

        if (isDelay) {
            predictionText.innerHTML = `🚨 HIGH DELAY RISK: ${percentage}% probability`;
            resultDiv.className = 'result delay';
            
            // Provide recommendations
            confidenceDiv.innerHTML = `
                <strong>Recommended Actions:</strong>
                <ul style="text-align: left; margin-top: 10px; font-size: 1rem;">
                    <li>Consider alternative routes</li>
                    <li>Check vehicle maintenance status</li>
                    <li>Notify customer about potential delay</li>
                    <li>Allocate additional resources</li>
                </ul>
            `;
        } else {
            predictionText.innerHTML = `✅ LOW DELAY RISK: ${percentage}% probability`;
            resultDiv.className = 'result no-delay';
            confidenceDiv.innerHTML = `<strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%`;
        }

        resultDiv.style.display = 'block';
        
        // Add animation
        resultDiv.style.animation = 'none';
        setTimeout(() => {
            resultDiv.style.animation = 'fadeIn 0.5s ease-in';
        }, 10);
    }
}

// Global instance
const predictor = new LogisticsPredictor();

// Global prediction function
async function predictDelay() {
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    
    // Show loading
    loadingDiv.style.display = 'block';
    resultDiv.style.display = 'none';
    
    try {
        // Get current input values
        const currentInput = LogisticsData.getCurrentInput();
        
        // Simulate model prediction (replace with actual model call)
        const delayProbability = await simulatePrediction(currentInput);
        const confidence = 0.85 + (Math.random() * 0.1); // Simulated confidence
        
        // Display result
        predictor.displayResult(delayProbability, confidence);
        
    } catch (error) {
        console.error('Prediction error:', error);
        document.getElementById('predictionText').innerHTML = 
            '❌ Error making prediction. Please try again.';
        document.getElementById('result').className = 'result delay';
        document.getElementById('result').style.display = 'block';
    } finally {
        loadingDiv.style.display = 'none';
    }
}

// Simulate prediction (replace with actual model in production)
async function simulatePrediction(inputData) {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Simple heuristic-based prediction for demo
    let probability = 0.3; // Base probability
    
    // Increase probability based on risk factors
    if (inputData[3] === 1) probability += 0.3; // Shipment status: Delayed
    if (inputData[6] === 1) probability += 0.2; // Traffic: Heavy
    if (inputData[6] === 2) probability += 0.25; // Traffic: Detour
    if (inputData[7] > 45) probability += 0.15; // Long waiting time
    if (inputData[10] !== 0) probability += 0.2; // Has delay reason
    if (inputData[4] < 10 || inputData[4] > 35) probability += 0.1; // Extreme temperature
    
    // Ensure probability is between 0 and 1
    return Math.min(Math.max(probability, 0.1), 0.95);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Smart Logistics Predictor initialized');
    
    // Add some sample data cycling for demo
    setInterval(() => {
        if (Math.random() > 0.7) {
            const inventories = [280, 320, 350, 400, 450];
            const temps = [22, 25, 28, 30, 18];
            
            document.getElementById('inventory').value = 
                inventories[Math.floor(Math.random() * inventories.length)];
            document.getElementById('temperature').value = 
                temps[Math.floor(Math.random() * temps.length)];
        }
    }, 5000);
});
