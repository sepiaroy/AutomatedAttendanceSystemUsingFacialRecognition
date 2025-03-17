Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "http://127.0.0.1:5000/classify_image", // Use correct backend URL
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

    dz.on("addedfile", function() {
        if (dz.files.length > 1) {
            dz.removeFile(dz.files[0]);        
        }
    });

    $("#submitBtn").on('click', function () {
        if (dz.files.length > 0) {
            var file = dz.files[0];
            var reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = function() {
                var url = "http://127.0.0.1:5000/classify_image";
                $.post(url, { image_data: reader.result }, function(data) {
                    console.log(data);
                    if (!data || data.length === 0) {
                        alert("Cannot classify image");
                    } else {
                        alert("Attendance marked!");
                    }
                });
            };
        }
    });
}

$(document).ready(function() {
    console.log("ready!");
    init();
});
