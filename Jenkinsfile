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
  
// -------------------------------------------------------------------------- //
    stage("Preparation") {
      parallel {
      
        stage("Prepare Linux") {
          agent { label 'ubuntu' }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            sh '''#!/bin/bash
              eval "$(/builds/miniconda3/bin/conda shell.bash hook)"
              echo "WD=$(pwd)"
              python3 -m venv --clear .test_venv
              source .test_venv/bin/activate
              python -m pip install -U pip pytest
            '''
          }
        }
        
        stage("Prepare MacOS") {
          agent { label 'macos' }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            sh '''#!/bin/bash
              echo "WD=$(pwd)"
              python3 -m venv --clear .test_venv
              source .test_venv/bin/activate
              python -m pip install -U pip pytest
            '''
          }
        }
        
        stage("Prepare Cuda") {
          agent { label 'cuda' }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            sh '''#!/bin/bash
              echo "WD=$(pwd)"
              python3 -m venv --clear .test_venv
              source .test_venv/bin/activate
              python -m pip install -U pip pytest
            '''
          }
        }
      }
    }

// -------------------------------------------------------------------------- //
    stage('Test Jenkins CI') {
      parallel {

        stage('Check Linux config') {
          agent { label 'ubuntu' }
          steps {
            echo 'Testing...'
            sh '''#!/bin/bash
              echo "WD=$(pwd)"
              source .test_venv/bin/activate
              echo "Python path = $(which python)"
              echo "Python version = $(python -V)"
            '''
          }
        }
        
        stage('Check MacOs config') {
          agent { label 'macos' }
          steps {
            echo 'Testing...'
            sh '''#!/bin/bash
              echo "WD=$(pwd)"
              source .test_venv/bin/activate
              echo "Python path = $(which python)"
              echo "Python version = $(python -V)"
            '''
          }
        }
        
        stage('Check Cuda config') {
          agent { label 'cuda' }
          steps {
            echo 'Testing...'
            sh '''#!/bin/bash
              echo "WD=$(pwd)"
              source .test_venv/bin/activate
              echo "Python path = $(which python)"
              echo "Python version = $(python -V)"
            '''
          }
        }
      }
    }

// -------------------------------------------------------------------------- //

/* Skipping PyKeOps test because of Jenkinsfile rewriting in progress
    stage('Test PyKeOps') {
      parallel {

        stage('Test Linux') {
          agent { label 'ubuntu' }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            echo 'Testing...'
            sh '''#!/bin/bash
              eval "$(/builds/miniconda3/bin/conda shell.bash hook)"
              source .test_venv/bin/activate
              python -m pip install ./keopscore
              python -m pip install ./pykeops
              pytest -v pykeops/pykeops/test/
            '''
          }
        }

        stage('Test Mac') {
          agent { label 'macos' }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            echo 'Testing...'
            sh '''
              source .test_venv/bin/activate
              python -m pip install ./keopscore
              python -m pip install ./pykeops
              pytest -v pykeops/pykeops/test/
            '''
          }
        }

        stage('Test Cuda') {
          agent { label 'cuda' }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            echo 'Testing..'
            sh '''#!/bin/bash
              source .test_venv/bin/activate
              python -m pip install ./keopscore
              python -m pip install ./pykeops
              pytest -v pykeops/pykeops/test/
            '''
          }
        }
      }
    }
*/

// -------------------------------------------------------------------------- //

/* Skipping RKeOps because not available yet in python_engine
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
*/

// ----------------------------------------------------------------------------------------

/* Skipping KeOpsLab because not available yet in python_engine
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
*/

/* skipping Doc stage because Oban computer is currently down as of march 9th 2021 - Joan
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
*/
/*
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
           cd pykeops/pykeops
           sh ./generate_wheel.sh -v ${TAG_NAME##v}
        '''
        withCredentials([usernamePassword(credentialsId: '8c7c609b-aa5e-4845-89bb-6db566236ca7', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'cd build && twine upload --repository-url https://test.pypi.org/legacy/ -u ${USERNAME} -p ${PASSWORD} ./dist/pykeops-${TAG_NAME##v}.tar.gz'
        }
      }
    }
    */

  }
}

