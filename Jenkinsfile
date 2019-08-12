// IC Script for Keops
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
          environment { CXX="g++-8" }
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
          environment { PATH="/Users/ci/miniconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin" }
          steps {
            echo 'Testing...'
              sh 'git submodule update --init'
              sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_pytorch.py'
              sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_numpy.py'
          }
        }

        stage('Test Cuda') {
          agent { label 'cuda' }
          environment { 
            CXX="g++-8"
            // PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/jenkins/.local/bin/"
          }
          steps {
            echo 'Testing..'
              sh 'git submodule update --init'
              sh '''
                 . /opt/miniconda3/bin/activate keops
                 cd pykeops/test
                 python unit_tests_pytorch.py
              '''
              sh '''
                 . /opt/miniconda3/bin/activate keops
                 cd pykeops/test
                 python unit_tests_numpy.py
              '''
          }
        }
      }
    }

// ----------------------------------------------------------------------------------------
    stage('Test KeOpsLab') {
      parallel {

        stage('Test Cuda') {
          agent { label 'matlab' }
          environment { 
            PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/jenkins/.local/bin/"
          }
          steps {
            echo 'Testing..'
              sh 'git submodule update --init'
              sh '''
                 cd keopslab/test
                 export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
                 matlab -nodisplay -r "r=runtests(\'generic_test.m\'),exit(sum([r(:).Failed]))"
              '''
          }
        }

      }
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
          . /opt/miniconda3/bin/activate keops
          cd doc/
          sh ./generate_doc.sh -b -l -v ${TAG_NAME##v}
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
      environment { PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-7.5/bin:/home/jenkins/.local/bin/" }
      steps {
        echo 'Deploying on tag event...'
        sh 'git submodule update --init'
        echo 'Deploying the wheel package...'
        sh 'cd pykeops && sh ./generate_wheel.sh -v ${TAG_NAME##v}'
        withCredentials([usernamePassword(credentialsId: '8c7c609b-aa5e-4845-89bb-6db566236ca7', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'cd build && twine upload --repository-url https://test.pypi.org/legacy/ -u ${USERNAME} -p ${PASSWORD} ./dist/pykeops-${TAG_NAME##v}.tar.gz'
          }
      }
    }

  }
}
