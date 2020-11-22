// IC Script for Keops

node {
  checkout scm
  result = sh (script: "git log -1 | grep '\\[ci skip\\]'", returnStatus: true)
  if (result == 0) {
    currentBuild.result = 'ABORTED'
    error('Detect [no ci] message in commit message. Not running.')
  }
}

pipeline {
  agent none 
  stages {

// ----------------------------------------------------------------------------------------
    stage('Test KeOps++') {
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
              sh 'cd keops/build && make -j15 VERBOSE=0'
          }
        }

      }
    }


// ----------------------------------------------------------------------------------------
    stage('Test PyKeOps') {
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
          steps {
            echo 'Testing...'
              sh 'git submodule update --init'
              sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_pytorch.py'
              sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_numpy.py'
          }
        }

        stage('Test Cuda') {
          agent { label 'cuda' }
          steps {
            echo 'Testing..'
              sh 'git submodule update --init'
              sh '''#!/bin/bash
                 eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
                 conda activate keops
                 cd pykeops/test
                 python unit_tests_pytorch.py
              '''
              sh '''#!/bin/bash
                 eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
                 conda activate keops
                 cd pykeops/test
                 python unit_tests_numpy.py
              '''
          }
        }

      }
    }

// ----------------------------------------------------------------------------------------
    stage('Test RKeOps') {
      parallel {

        stage('Test Linux') {
          agent { label 'ubuntu' }
          steps {
            echo 'Testing..'
              sh 'git submodule update --init'
              sh '''
                 bash rkeops/ci/run_ci.sh
              '''
          }
        }

        stage('Test Mac') {
          agent { label 'macos' }
          steps {
            echo 'Testing..'
              sh 'git submodule update --init'
              sh '''
                 # bash rkeops/ci/run_ci.sh
              '''
          }
        }

        stage('Test Cuda') {
          agent { label 'cuda' }
          steps {
            echo 'Testing..'
              sh 'git submodule update --init'
              sh '''
                 export TEST_GPU=1
                 bash rkeops/ci/run_ci.sh
              '''
          }
        }

      }
    }

// ----------------------------------------------------------------------------------------
    stage('Test KeOpsLab') {
      //parallel {

        //stage('Test Cuda') {
          agent { label 'matlab' }
          steps {
            echo 'Testing..'
              sh 'git submodule update --init'
              sh '''
                 cd keopslab/test
                 export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
                 matlab -nodisplay -r "r=runtests('generic_test.m'),exit(sum([r(:).Failed]))"
              '''
          }
        //}

      //}
    }



// ----------------------------------------------------------------------------------------
    stage('Doc') {
      when { buildingTag() }
      agent { label 'cuda-doc' }
      steps {
        echo 'Generating doc on tag event...'
        sh 'git submodule update --init'
        echo 'Building the doc...'
        sh '''
          eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
          conda activate keops
          cd doc/
          sh ./generate_doc.sh -b -l -v ${TAG_NAME##v} -n 2
        '''
        withCredentials([usernamePassword(credentialsId: '02af275f-5383-4be3-91d8-4c711aa90de9', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh '''
             lftp -u ${USERNAME},${PASSWORD}  -e "mirror -e -R  ./doc/_build/html/ /www/keops_latest/ ; quit" ftp://ftp.cluster021.hosting.ovh.net
          '''
        }
      }
    }

// ----------------------------------------------------------------------------------------
    stage('Deploy') {
      when { buildingTag() }
      agent { label 'cuda' }
      steps {
        echo 'Deploying on tag event...'
        sh 'git submodule update --init'
        echo 'Deploying the wheel package...'
        sh '''
           eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
           conda activate keops
           cd pykeops
           sh ./generate_wheel.sh -v ${TAG_NAME##v}
        '''
        withCredentials([usernamePassword(credentialsId: '8c7c609b-aa5e-4845-89bb-6db566236ca7', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'cd build && twine upload --repository-url https://test.pypi.org/legacy/ -u ${USERNAME} -p ${PASSWORD} ./dist/pykeops-${TAG_NAME##v}.tar.gz'
          }
      }
    }

  }
}

