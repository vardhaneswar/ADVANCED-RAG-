pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo "Cloning code from GitHub..."
                git branch: 'main',
                    credentialsId: 'github-credentials',
                    url: 'https://github.com/vardhaneswar/ADVANCED-RAG-.git'
            }
        }
        stage('Build') {
            steps {
                echo "Installing dependencies..."
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r services/rag-api-service/requirements.txt
                '''
            }
        }
    }
}