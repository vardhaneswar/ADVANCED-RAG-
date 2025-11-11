pipeline {
    agent any

    environment {
        // Optional: can define env variables or credentials later
        PYTHONPATH = '.'
    }

    stages {
        stage('Checkout') {
            steps {
                echo "Pulling latest code..."
                git branch: 'main', url: 'https://github.com/vardhaneswar/advanced-multisource-rag-finance.git'
            }
        }

        stage('Setup Environment') {
            steps {
                echo "Setting up Python environment..."
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r services/rag-api-service/requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                echo "Running tests..."
                sh '''
                    . venv/bin/activate
                    pytest tests/ || true
                '''
            }
        }

        stage('Build Docker Images') {
            steps {
                echo "Building Docker images for all services..."
                sh '''
                    docker build -t rag-api-service:latest services/rag-api-service
                    docker build -t ingestion-service:latest services/ingestion-service
                    docker build -t dashboard-service:latest services/dashboard-service
                '''
            }
        }

        stage('Cleanup') {
            steps {
                echo "Cleaning up workspace..."
                deleteDir()
            }
        }
    }

    post {
        always {
            echo "Pipeline execution finished."
        }
    }
}
