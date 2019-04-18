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
          sh 'git submodule update --init'
          sh 'cd pykeops/test && python3 unit_tests_pytorch.py'
          sh 'cd pykeops/test && python3 unit_tests_numpy.py'
      }
    }
  }
  agent {
    label 'macos'
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
          sh 'git submodule update --init'
          sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_pytorch.py'
          sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_numpy.py'
      }
    }
  }
}
        
