@Library('ecdc-pipeline')
import ecdcpipeline.ContainerBuildNode
import ecdcpipeline.PipelineBuilder

project = "nexus-constructor"

// Set number of old artefacts to keep.
properties([
    buildDiscarder(
        logRotator(
            artifactDaysToKeepStr: '',
            artifactNumToKeepStr: '5',
            daysToKeepStr: '',
            numToKeepStr: ''
        )
    )
])

container_build_nodes = [
  'centos7': ContainerBuildNode.getDefaultContainerBuildNode('centos7-gcc8')
]

pipeline_builder = new PipelineBuilder(this, container_build_nodes)

builders = pipeline_builder.createBuilders { container ->

    pipeline_builder.stage("Checkout") {
        dir(pipeline_builder.project) {
            scm_vars = checkout scm
        }
        // Copy source code to container
        container.copyTo(pipeline_builder.project, pipeline_builder.project)
    }  // stage

    pipeline_builder.stage("Create virtualenv") {
        container.sh """
            cd ${project}
            python3.6 -m venv build_env
        """
    } // stage

    pipeline_builder.stage("Install requirements") {
        container.sh """
            cd ${project}
            build_env/bin/pip --proxy ${https_proxy} install --upgrade pip
            build_env/bin/pip --proxy ${https_proxy} install -r requirements-base.txt
            build_env/bin/pip --proxy ${https_proxy} install -r requirements.txt
            """
    } // stage

    pipeline_builder.stage("Run tests") {
        def testsError = null
        try {
                container.sh """
                    cd ${project}
                    build_env/bin/python -m pytest -s ./examples/tests --ignore=build_env
                """
            }
            catch(err) {
                testsError = err
                currentBuild.result = 'FAILURE'
            }

    } // stage
}

def get_macos_pipeline() {
    return {
        node('macos') {
            cleanWs()
            dir("${project}") {
                stage('Checkout') {
                    try {
                        checkout scm
                    } catch (e) {
                        failure_function(e, 'MacOSX / Checkout failed')
                    } // catch
                } // stage
                stage('Setup') {
                    sh """
                        mkdir -p ~/virtualenvs
                        /opt/local/bin/python3.6 -m venv ~/virtualenvs/${pipeline_builder.project}-${pipeline_builder.branch}
                        source ~/virtualenvs/${pipeline_builder.project}-${pipeline_builder.branch}/bin/activate
                        pip --proxy=${https_proxy} install --upgrade pip
                        pip --proxy=${https_proxy} install -r requirements-base.txt
                        pip --proxy=${https_proxy} install -r requirements.txt
                    """
                } // stage
                stage('Run tests') {
                    sh """
                        source ~/virtualenvs/${pipeline_builder.project}-${pipeline_builder.branch}/bin/activate
                        python -m pytest . -s
                    """
                } // stage
            } // dir
        } // node
    } // return
} // def

node("docker") {
    cleanWs()

    stage('Checkout') {
        dir("${project}") {
            try {
                scm_vars = checkout scm
            } catch (e) {
                failure_function(e, 'Checkout failed')
            }
        }
    }

    builders['macOS'] = get_macos_pipeline()
    parallel builders
}
