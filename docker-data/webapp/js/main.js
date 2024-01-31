const EXPERIMENTATION_SERVER_URL = "http://localhost:5253";

const VID_WIDTH = 1280, VID_HEIGHT = 720;
const HIDDEN_CANVAS_WIDTH = 320, HIDDEN_CANVAS_HEIGHT = 180;
// const HIDDEN_CANVAS_WIDTH = 640, HIDDEN_CANVAS_HEIGHT = 360;

let id_calibrated_model = '', id_in_progress_calibration;

let sock;
let video_origin, canvas_origin;


let has_recently_updated_data = false;
let PREDICTION_TIMEOUT = 500;

let user_id = 1;
let isTakingSnapshot = false;

function init() {
    // MEDIA WEBCAM CAPTURE
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
        alert("Your browser doesn't seem to support the use of a webcam. Please use a more modern browser.");
        return;
    }

    video_origin = document.createElement('video');
    video_origin.id = 'video_origin';
    video_origin.width = VID_WIDTH;
    video_origin.height = VID_HEIGHT;

    canvas_origin = document.createElement('canvas');
    canvas_origin.width = HIDDEN_CANVAS_WIDTH;
    canvas_origin.height = HIDDEN_CANVAS_HEIGHT;

    navigator.mediaDevices.getUserMedia({
            video: true
        })
        .then(stream => {
            video_origin.srcObject = stream;
            video_origin.onloadedmetadata = (e) => video_origin.play();
        })
        .catch(msg => console.log('Error: ' + msg));


    // SOCKET.IO
    sock = io.connect('http://' + document.domain + ':' + location.port);

    sock.on('connect',
        function() {
            console.log('Initialised SocketIO connection...');

            // START CAPTURE
            capture();
        });

    sock.on('disconnect',
        function() {
            console.log('Terminated SocketIO connection.');
        });

    sock.on('update-data', (data) =>
        {
            // console.log(data);

            if (!has_recently_updated_data) {
                has_recently_updated_data = true;

                data.forEach(face => {
                    // !! NOTE: Remove the line below to display the information of the last face instead.
                    // !! TODO: Create a multi-face information display
                    face = data[0]

                    face.forEach(emotion => {
                        prediction_text = (emotion[1]*100).toFixed(2) + '%';

                        switch (emotion[0]) {
                            case 'angry':
                                data_angry.innerText = prediction_text;
                                break;
                            case 'disgust':
                                data_disgust.innerText = prediction_text;
                                break;
                            case 'fear':
                                data_fear.innerText = prediction_text;
                                break;
                            case 'happy':
                                data_happy.innerText = prediction_text;
                                break;
                            case 'neutral':
                                data_neutral.innerText = prediction_text;
                                break;
                            case 'sad':
                                data_sad.innerText = prediction_text;
                                break;
                            case 'surprise':
                                data_surprise.innerText = prediction_text;
                                break;
                        }
                    });
                });

                setTimeout(function () {
                    has_recently_updated_data = false;
                }, PREDICTION_TIMEOUT)
            }
        });

    // Disconnect before closing the window
    window.onunload = function() {
        sock.disconnect()
    }
}
// CAPTURE AND MANIPULATE WEBCAM FEED
const capture = () => {
    if (isTakingSnapshot) {
        // Stop capturing frames when taking a snapshot
        return;
    }

    canvas_origin.getContext('2d').drawImage(video_origin, 0, 0, canvas_origin.width, canvas_origin.height);
    canvas_origin.toBlob((blob) => {
        current_frame = blob;

        sock.emit('process-image', {user_id: id_calibrated_model, image_blob: blob}, (data) => {
            let imgData = new Blob([data], {type: 'image/jpg'});
            let img = new Image();
            img.onload = () => preview.getContext('2d').drawImage(img, 0, 0, preview.width, preview.height);
            img.src = URL.createObjectURL(imgData);

            if (!isTakingSnapshot) {
                // Continue capturing frames unless taking a snapshot
                capture();
            }
        });
    });
}

function changeUserId() {
    var newUserId = document.getElementById("userIdInput").value;
    if (newUserId !== "") {
        // Update the user_id variable
        user_id = newUserId;
        alert("User ID changed to: " + user_id);
    } else {
        alert("Please enter a valid User ID.");
    }
}


function uploadDataset() {
    // Experimentation URL
    let url = EXPERIMENTATION_SERVER_URL + `/upload/dataset?user_id=${user_id}`;
    console.log(`uploading for id: ${user_id}`);

    // Get zip file and add it to the form data
    let formData = new FormData();
    formData.append('zipfile', input_dataset_upload.files[0], input_dataset_upload.files[0].name);

    // Send request
    $.ajax({
        url: url,
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        success: function(result) {
            console.log(`Successfully uploaded dataset.`);
            window.alert(`Successfully uploaded dataset.`);
        },
        error: function(error) {
            console.log(`Error in uploading dataset: ${error}`);
            window.alert('Could not upload the dataset at this time. Please try again later.');
        },
    });
}

function downloadDataset() {

    let url = EXPERIMENTATION_SERVER_URL + `/download/dataset?user_id=${user_id}`;
    console.log(`Downloading for id: ${user_id}`);

    // Make a fetch request to trigger the dataset download
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.blob();
        })
        .then(blob => {
            // Create a link element
            const link = document.createElement('a');

            // Create a Blob URL from the response blob
            const url = window.URL.createObjectURL(blob);

            // Set the link's href to the Blob URL
            link.href = url;

            // Set the link's download attribute to the desired file name
            link.download = 'dataset.zip';

            // Append the link to the document
            document.body.appendChild(link);

            // Trigger a click event on the link to start the download
            link.click();

            // Remove the link from the document
            document.body.removeChild(link);

            // Revoke the Blob URL to free up resources
            window.URL.revokeObjectURL(url);
        })
        .catch(error => console.error('Fetch error:', error));
}


function takeSnapshot() {
    // Set the flag to stop capturing frames
    isTakingSnapshot = true;

    // Capture the current frame
    canvas_origin.getContext('2d').drawImage(video_origin, 0, 0, canvas_origin.width, canvas_origin.height);
    canvas_origin.toBlob((blob) => {
        current_frame = blob;

        // Display the snapshot
        let imgData = new Blob([current_frame], {type: 'image/jpg'});
        let img = new Image();
        img.onload = () => preview.getContext('2d').drawImage(img, 0, 0, preview.width, preview.height);
        img.src = URL.createObjectURL(imgData);
    });
}
    const toggleBtn = document.querySelector('.model-snapshot-upload')
    const selectBtn = document.querySelector(".select-btn"),
        items = document.querySelectorAll(".item");

    toggleBtn.addEventListener("click", () => {
            selectBtn.classList.toggle("active");
            });
    selectBtn.addEventListener("click", () => {
        selectBtn.classList.toggle("open");

        items.forEach(item => {
            item.addEventListener("click", () => {
                item.classList.toggle("checked");

                let checked = document.querySelectorAll(".checked"),
                    btnText = document.querySelector(".btn-text");

                if (checked && checked.length > 0) {
                    btnText.innerText = `${checked.length} Selected`;
                } else {
                    btnText.innerText = "Select Correct Emotion";
                }
            });
        });
    });
    

function uploadSnapshot() {
    // Check if snapshot has been made
    if (isTakingSnapshot) {
        let correction = document.querySelector(".checked");
        let emotion = correction.textContent.trim().toLowerCase();

        // Convert the displayed snapshot to Blob
        preview.toBlob((blob) => {
            var formData = new FormData();
            formData.append('snapshot', blob);

            if (!correction) {
                console.log(`Error in emotion selection`);
                window.alert('No emotion selected.');
            } else {
                let url = EXPERIMENTATION_SERVER_URL + `/snapshot/upload?user_id=${user_id}&emotion=${emotion}`;

                // Send the captured frame to the server for processing or storage
                $.ajax({
                    url: url,
                    type: 'POST',
                    data: formData,
                    contentType: false,
                    processData: false,
                    success: function (result) {
                        console.log(`Successfully uploaded snapshot.`);
                        window.alert('Successfully uploaded snapshot.');
                    },
                    error: function (error) {
                        console.log(`Error in uploading snapshot: ${error}`);
                        window.alert('Could not upload snapshot at this time. Please try again later.');
                    },
                });

                // Reset the flag to resume capturing frames
                isTakingSnapshot = false;
                capture();
            }
        });
    } else {
        console.log('Take a snapshot first.');
        window.alert('Take a snapshot first.');
    }
}

    

function runPipeline() {
    // Get zip file and add it to the form data
    let formData = new FormData();
    formData.append('zipfile', input_dataset_train.files[0], input_dataset_train.files[0].name);

    let url = EXPERIMENTATION_SERVER_URL + `/snapshot/train?user_id=${user_id}`;
    // Call training on experimentation server.
    $.ajax({
        url: url,
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        success: function(result) {
            console.log('Training completed.');
            window.alert('A training procedure has been completed. A new best model has been placed.');
        },
        error: function(error) {
            console.log(`Error in training call: ${error}`);
            window.alert('Could not complete a training procedure at this time. Please try again later.');
        },
    });
    console.log('Training initiated.');
    window.alert('Training has been started. When the process has finished, it will ' +
        'automatically be applied to this current session. You will be notified when the personalised model is ready.');
}


function resume() {
    isTakingSnapshot = false;
    capture();
}