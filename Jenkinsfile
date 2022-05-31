pipeline {
  agent none
  stages {
    stage('Test PyKeOps') {
      parallel {
        stage('Test Linux') {
          agent {
            label 'ubuntu'
          }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            echo 'Testing...'
            sh '''#!/bin/bash
              eval "$(/builds/miniconda3/bin/conda shell.bash hook)"
              bash pytest.sh -v
            '''
          }
        }

        stage('Test Mac') {
          agent {
            label 'macos'
          }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            echo 'Testing...'
            sh '''
              bash pytest.sh -v
            '''
          }
        }

        stage('Test Cuda') {
          agent {
            label 'cuda'
          }
          steps {
            echo 'Clean KeOps Cache...'
            sh 'rm -rf $HOME/.cache/keops*'
            echo 'Testing..'
            sh '''#!/bin/bash
              srun -n 1 -c 16 --mem=8G --gpus=1 --gres-flags=enforce-binding               -J keops_ci pytest.sh -v
            '''
          }
        }

      }
    }

  }
}