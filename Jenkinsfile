pipeline {
  agent {
    label 'ubuntu'
  }
    stages {
      stage('Build') {
        steps {
          echo 'Building..'
          sh 'git submodule update --init'
          sh 'cd keops/build && cmake ..'
          sh 'cd keops/build && make VERBOSE=0'
        }
      }
      stage('Test') {
        steps {
          echo 'Testing..'
      }
    }
  }
}
        
