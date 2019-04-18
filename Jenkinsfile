pipeline {
  agent none 
    stages {
      stage('Build') {
        parallel {
          stage('Build in Linux') {
            agent { label 'ubuntu' }
            steps {
              echo 'Building..'
                sh 'git submodule update --init'
                sh 'cd keops/build && cmake ..'
                sh 'cd keops/build && make VERBOSE=0'
            }
          }
          stage('Build Mac') {
            agent { label 'macos' }
            steps {
              echo 'Building..'
                sh 'git submodule update --init'
                sh 'cd keops/build && cmake ..'
                sh 'cd keops/build && make VERBOSE=0'
            }
          }
           stage('Build Cuda') {
             agent { label 'cuda' }
             steps {
              echo 'Building..'
                sh 'git submodule update --init'
                sh 'cd keops/build && cmake ..'
                sh 'cd keops/build && make VERBOSE=0'
             }
           }
        }
      }
      stage('Test') {
        parallel {
          stage('Test Linux') {
            agent { label 'ubuntu' }
            steps {
              echo 'Testing..'
                sh 'git submodule update --init'
                sh 'cd pykeops/test && python3 unit_tests_pytorch.py'
                sh 'cd pykeops/test && python3 unit_tests_numpy.py'
            }
          }
          stage('Test Mac') {
            agent { label 'macos' }
            environment { PATH="/Users/ci/miniconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin" }
            steps {
              echo 'Testing..'
                sh 'git submodule update --init'
                sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_pytorch.py'
                sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_numpy.py'
            }
          }
        }
      }
    }
}
