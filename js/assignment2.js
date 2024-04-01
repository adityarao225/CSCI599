import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

function initScene(containerId, plyPath) {
    const container = document.getElementById(containerId);
    container.style.position = 'relative';

    let renderer, stats, gui;
    let scene, camera, controls, pointCloud, dirlight, ambientLight;
    let isinitialized = false;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.minDistance = 2;
    controls.maxDistance = 10;
    controls.addEventListener('change', function () { renderer.render(scene, camera); });

    dirlight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirlight.position.set(0, 0, 1);
    scene.add(dirlight);

    ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    let loader = new PLYLoader();
    loader.load(
        plyPath,
        function (geometry) {
            geometry.computeVertexNormals(); // Important if the PLY contains vertex colors or normals
            const material = new THREE.PointsMaterial({ size: 0.05, vertexColors: true });
            pointCloud = new THREE.Points(geometry, material);
            pointCloud.name = "pointCloud";
            scene.add(pointCloud);
        },
        function (xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        function (error) {
            console.log('An error happened: ' + error);
        }
    );

    camera.position.z = 5;

    function initStats() {
        stats = new Stats();
        stats.showPanel(0);
        stats.dom.style.position = 'absolute';
        stats.dom.style.top = 0;
        stats.dom.style.left = 0;
        container.appendChild(stats.dom);
    }

    function initGUI() {
        if (!isinitialized && pointCloud) {
            gui = new GUI();
            gui.add(pointCloud.position, 'x', -1, 1);
            gui.add(pointCloud.position, 'y', -1, 1);
            gui.add(pointCloud.position, 'z', -1, 1);
            gui.domElement.style.position = 'absolute';
            gui.domElement.style.top = '0px';
            gui.domElement.style.right = '0px';
            container.appendChild(gui.domElement);
            isinitialized = true;
        }
    }

    function animate() {
        requestAnimationFrame(animate);

        pointCloud = scene.getObjectByName("pointCloud");
        if (pointCloud) {
            pointCloud.rotation.x += 0.01;
            pointCloud.rotation.y += 0.01;
            initGUI();
        }

        renderer.render(scene, camera);
        stats.update();
    }

    function onWindowResize() {
        camera.aspect = window.innerWidth / (window.innerHeight * 0.5);
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
    }

    window.addEventListener('resize', onWindowResize, false);

    initStats();
    animate();
}

// Initialize the scenes with .ply files
initScene('container1', '../assets/assignment2/results/fountain-P11/point-clouds/cloud_11_view.ply');
initScene('container2', '../plots/colmap_fountain.ply');
initScene('container3', '../assets/assignment2/results/Herz-Jesus-P8/point-clouds/cloud_8_view.ply');
initScene('container4', '../plots/colmap_Jesus.ply');

