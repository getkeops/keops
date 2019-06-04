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
            //sh 'cd keops/build && cmake ..'
            //sh 'cd keops/build && make VERBOSE=0'
          }
        }
        
        stage('Build Mac') {
          agent { label 'macos' }
          steps {
            echo 'Building..'
            sh 'git submodule update --init'
            //sh 'cd keops/build && cmake ..'
            //sh 'cd keops/build && make VERBOSE=0'
          }
        }
        
        stage('Build Cuda') {
          agent { label 'cuda' }
          steps {
            echo 'Building..'
            sh 'git submodule update --init'
            //sh 'cd keops/build && cmake ..'
            //sh 'cd keops/build && make VERBOSE=0'
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
            //sh 'cd pykeops/test && python3 unit_tests_pytorch.py'
            //sh 'cd pykeops/test && python3 unit_tests_numpy.py'
          }
        }
        
        stage('Test Mac') {
          agent { label 'macos' }
          environment { PATH="/Users/ci/miniconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin" }
          steps {
            echo 'Testing...'
            sh 'git submodule update --init'
            //sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_pytorch.py'
            //sh 'cd pykeops/test && /Users/ci/miniconda3/bin/python3 unit_tests_numpy.py'
          }
        }
        
        stage('Test Cuda') {
          agent { label 'cuda' }
          steps {
            echo 'Testing..'
            sh 'git submodule update --init'
            //sh 'cd pykeops/test && python3 unit_tests_pytorch.py'
            //sh 'cd pykeops/test && python3 unit_tests_numpy.py'
            //sh '''
            //    cd keopslab/test
            //    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
            //    matlab -nodisplay -r "r=runtests(\'generic_test.m\'),exit(sum([r(:).Failed]))"
            //'''
            echo 'Deploying on tag event...'
            sh 'git submodule update --init'
            echo 'Building the doc...'
            sh 'cd doc/ && sh ./generate_doc.sh'
            echo 'Deploying the wheel package...'
            sh 'cd pykeops && sh ./generate_wheel -v ${TAG_NAME##v}'
            withCredentials([usernamePassword(credentialsId: '8c7c609b-aa5e-4845-89bb-6db566236ca7', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
              sh 'set +x && cd build && twine upload -u ${USERNAME} -p ${PASSWORD} ../build/dist/pykeops-${TAG_NAME##v}.tar.gz'
            }
          }
        }
      }
    }


    stage('Deploy') {
      agent { label 'cuda' }
      when { buildingTag() }
      steps {
        echo 'Deploying on tag event...'
        sh 'git submodule update --init'
        echo 'Building the doc...'
        sh 'cd doc/ && sh ./generate_doc.sh'
        echo 'Deploying the wheel package...'
        sh 'cd pykeops && sh ./generate_wheel -v ${TAG_NAME##v}'
        withCredentials([usernamePassword(credentialsId: '8c7c609b-aa5e-4845-89bb-6db566236ca7', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'set +x && cd build && twine upload -u ${USERNAME} -p ${PASSWORD} ../build/dist/pykeops-${TAG_NAME##v}.tar.gz'
        }
      }
    }

  }
}
