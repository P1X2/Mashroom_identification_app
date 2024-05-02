var images = JSON.parse("{{ img_list_json|escapejs }}");
var currentImageIndex = 0;
var imageElement = document.getElementById("current-image");

if (Array.isArray(images) && images.length > 0) {
  imageElement.src = images[currentImageIndex];
} else {
  console.error("Nieprawidłowe dane zdjęć");
}

function nextImage() {
  currentImageIndex++;
  if (currentImageIndex >= images.length) {
    currentImageIndex = 0;
  }
  imageElement.src = images[currentImageIndex];
}

function previousImage() {
  currentImageIndex--;
  if (currentImageIndex < 0) {
    currentImageIndex = images.length - 1;
  }
  imageElement.src = images[currentImageIndex];
}

function changeTab(selector_side) {
  const selector = "." + selector_side + " .dropdown-content";
  var tab = document.querySelector(selector);

  // var rootStyles = window.getComputedStyle(document.documentElement);
  // var secondaryColor = rootStyles.getPropertyValue("--secondary-color");
  // var primaryColor = rootStyles.getPropertyValue("--primary-color");

    if (tab.style.display === "none") {
      tab.style.display = "block";
    }
    else {
      tab.style.display = "none";
    }
  }