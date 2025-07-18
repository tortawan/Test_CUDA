pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from your Git repository
                git 'https://github.com/tortawan/Test_CUDA.git'
            }
        }

        stage('Build') {
            steps {
                // Build the Docker image
                script {
                    docker.build("your-dockerhub-username/dqn-lunar-lander")
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                // Login to Docker Hub and push the image
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh "docker login -u ${DOCKER_USER} -p ${DOCKER_PASS}"
                    sh "docker push your-dockerhub-username/dqn-lunar-lander"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                // Apply the Kubernetes deployment configuration
                sh 'kubectl apply -f deployment.yaml'
            }
        }
    }
}

