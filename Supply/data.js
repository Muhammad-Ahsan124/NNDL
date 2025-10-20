// Sample data and model configuration
const LogisticsData = {
    // Feature scaling parameters (from training)
    featureMeans: [23.45, 15.67, 312.5, 1.2, 24.3, 65.8, 1.1, 35.2, 298.7, 5.8, 1.4, 78.3, 215.6],
    featureStds: [45.32, 85.41, 125.8, 0.8, 4.2, 12.3, 0.9, 15.6, 145.3, 3.2, 1.1, 15.7, 85.4],
    
    // Model configuration
    timeSteps: 10,
    featureCount: 13,
    
    // Sample historical data for sequence prediction
    getSampleSequence() {
        return [
            [40.7128, -74.0060, 350, 0, 25.0, 65.0, 0, 30, 300, 5, 0, 75.0, 250],
            [40.7130, -74.0058, 345, 0, 25.1, 65.2, 0, 28, 310, 5, 0, 76.0, 255],
            [40.7132, -74.0056, 340, 0, 25.2, 65.5, 0, 25, 320, 5, 0, 77.0, 260],
            [40.7134, -74.0054, 335, 0, 25.3, 65.8, 0, 22, 330, 5, 0, 78.0, 265],
            [40.7136, -74.0052, 330, 0, 25.4, 66.0, 0, 20, 340, 5, 0, 79.0, 270],
            [40.7138, -74.0050, 325, 0, 25.5, 66.2, 0, 18, 350, 5, 0, 80.0, 275],
            [40.7140, -74.0048, 320, 0, 25.6, 66.5, 0, 15, 360, 5, 0, 81.0, 280],
            [40.7142, -74.0046, 315, 0, 25.7, 66.8, 0, 12, 370, 5, 0, 82.0, 285],
            [40.7144, -74.0044, 310, 0, 25.8, 67.0, 0, 10, 380, 5, 0, 83.0, 290],
            [40.7146, -74.0042, 305, 0, 25.9, 67.2, 0, 8, 390, 5, 0, 84.0, 295]
        ];
    },
    
    // Scale features (z-score normalization)
    scaleFeatures(features) {
        return features.map((feature, index) => {
            return (feature - this.featureMeans[index]) / this.featureStds[index];
        });
    },
    
    // Prepare input sequence for LSTM
    prepareInput(currentFeatures) {
        const sequence = this.getSampleSequence();
        // Replace the last element with current features
        sequence[sequence.length - 1] = this.scaleFeatures(currentFeatures);
        
        // Scale all sequence elements
        return sequence.map(features => this.scaleFeatures(features));
    },
    
    // Get current input values from form
    getCurrentInput() {
        return [
            parseFloat(document.getElementById('latitude').value),
            parseFloat(document.getElementById('longitude').value),
            parseFloat(document.getElementById('inventory').value),
            parseInt(document.getElementById('shipmentStatus').value),
            parseFloat(document.getElementById('temperature').value),
            parseFloat(document.getElementById('humidity').value),
            parseInt(document.getElementById('trafficStatus').value),
            parseFloat(document.getElementById('waitingTime').value),
            parseFloat(document.getElementById('transactionAmount').value),
            parseFloat(document.getElementById('purchaseFrequency').value),
            parseInt(document.getElementById('delayReason').value),
            parseFloat(document.getElementById('assetUtilization').value),
            parseFloat(document.getElementById('demandForecast').value)
        ];
    }
};

// Performance metrics for dashboard
const PerformanceMetrics = {
    accuracy: 0.87,
    precision: 0.85,
    recall: 0.82,
    f1Score: 0.83,
    
    getMonthlyData() {
        return {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            delays: [45, 42, 38, 35, 32, 28, 25, 22, 20, 18, 16, 15],
            predictions: [42, 39, 36, 33, 30, 27, 24, 21, 19, 17, 15, 14]
        };
    }
};
