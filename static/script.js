let selectedFile = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    setupFileUpload();
});

function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.querySelector('.browse-btn');
    
    console.log('Setting up file upload...', { uploadArea, fileInput, browseBtn });
    
    // Handle click on upload area
    uploadArea.addEventListener('click', function(event) {
        console.log('Upload area clicked');
        // Don't trigger if clicking the browse button directly
        if (!event.target.classList.contains('browse-btn')) {
            fileInput.click();
        }
    });
    
    // Handle click on browse button
    if (browseBtn) {
        browseBtn.addEventListener('click', function(event) {
            console.log('Browse button clicked');
            event.stopPropagation();
            fileInput.click();
        });
    }
    
    // Handle file selection
    fileInput.addEventListener('change', function(event) {
        console.log('File input changed', event.target.files);
        if (event.target.files.length > 0) {
            handleFileSelect(event);
        }
    });
    
    // Handle drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById('uploadArea').classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById('uploadArea').classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById('uploadArea').classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPEG, PNG, or GIF)');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    // Only process if we don't already have this file
    if (selectedFile && selectedFile.name === file.name && selectedFile.size === file.size) {
        return; // Same file, don't process again
    }
    
    selectedFile = file;
    showFileInfo(file);
    enablePredictButton();
    hideError();
    hideResults();
}

function showFileInfo(file) {
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').style.display = 'block';
    document.getElementById('uploadArea').style.display = 'none';
}

function removeFile() {
    selectedFile = null;
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('fileInput').value = '';
    disablePredictButton();
    hideResults();
}

function enablePredictButton() {
    const btn = document.getElementById('predictBtn');
    btn.disabled = false;
}

function disablePredictButton() {
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
}

function showLoading() {
    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    
    btnText.textContent = 'Classifying...';
    loader.style.display = 'block';
    btn.disabled = true;
}

function hideLoading() {
    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    
    btnText.textContent = 'Classify Tumor';
    loader.style.display = 'none';
    btn.disabled = false;
}

async function predictImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    showLoading();
    hideError();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'An error occurred during prediction');
        }
    } catch (error) {
        showError('Network error. Please try again.');
        console.error('Error:', error);
    } finally {
        hideLoading();
    }
}

function displayResults(result) {
    // Show the uploaded image
    document.getElementById('previewImage').src = result.image;
    
    // Show main prediction
    document.getElementById('predictedClass').textContent = result.predicted_class;
    document.getElementById('confidence').textContent = 
        `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    
    // Show all probabilities
    displayProbabilities(result.all_predictions);
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

function displayProbabilities(predictions) {
    const container = document.getElementById('probabilityBars');
    container.innerHTML = '';
    
    // Sort predictions by probability
    const sortedPredictions = Object.entries(predictions)
        .sort(([,a], [,b]) => b - a);
    
    sortedPredictions.forEach(([className, probability]) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'probability-bar';
        
        barDiv.innerHTML = `
            <div class="probability-label">
                <span>${className}</span>
                <span>${(probability * 100).toFixed(1)}%</span>
            </div>
            <div class="probability-progress">
                <div class="probability-fill" style="width: ${probability * 100}%"></div>
            </div>
        `;
        
        container.appendChild(barDiv);
    });
}
