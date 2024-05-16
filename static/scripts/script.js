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

function changeTabLeft(selector_side, icon_id) {
  var tab_right = document.querySelector(".dropdown-content-right");
  if (tab_right.style.display != "none") {
    changeTabRight(".dropdown-content-right", "fa-user");
  }
  clickTabLeft(selector_side, icon_id);
}


function changeTabRight(selector_side, icon_id) {
  var tab_left = document.querySelector(".dropdown-content-left");
  if (tab_left.style.display != "none") {
    clickTabLeft(".dropdown-content-left", "fa-bars");
  }
  clickTabRight(selector_side, icon_id);
}

function clickTabRight(selector_side, icon_id) {
  var tab = document.querySelector(selector_side);
  var icon = document.getElementById("userIcon");
  var navbar = document.getElementById("navbarContainer");
  var button = document.querySelector(".dropdown-right .dropbtn");
  
  if (tab.style.display === "none") {
    navbar.style.width = "50%";
    tab.style.display = "flex";
    icon.classList.remove(icon_id);
    icon.classList.add('fa-times');
    button.style.backgroundColor = "var(--tab-color)";
  } else {
    navbar.style.width = "100%";
    tab.style.display = "none";
    icon.classList.remove("fa-times");
    icon.classList.add(icon_id);
    button.style.backgroundColor = "var(--primary-color)";
  }
}

function clickTabLeft(selector_side, icon_id) {
  var tab = document.querySelector(selector_side);
  var icon = document.getElementById("menuIcon");
  var navbar = document.getElementById("navbarContainer");
  var button = document.querySelector(".dropdown-left .dropbtn");
  if (tab.style.display === "none") {
    navbar.style.width = "50%";
    navbar.style.marginLeft = "50%";
    tab.style.display = "flex";
    icon.classList.remove(icon_id);
    icon.classList.add('fa-times');
    button.style.backgroundColor = "var(--tab-color)";
    // button.hovered.
  } else {
    navbar.style.width = "100%";
    navbar.style.marginLeft = "0%";
    tab.style.display = "none";
    icon.classList.remove("fa-times");
    icon.classList.add(icon_id);
    button.style.backgroundColor = "var(--primary-color)";
  }
}

function plusDivs(n) {
  showDivs(slideIndex += n);
}

function showDivs(n) {
  var i;
  var x = document.getElementsByClassName("mySlides");
  if (n > x.length) {slideIndex = 1}
  if (n < 1) {slideIndex = x.length} ;
  for (i = 0; i < x.length; i++) {
    x[i].style.display = "none";
  }
  x[slideIndex-1].style.display = "block";
}
