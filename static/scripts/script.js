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

function checkWidth() {
  const elemWidth = document.querySelector(".navbar-tab").offsetWidth;
  alert("Width of the div: " + elemWidth + " pixels");
}

function marginChange() {
  var tab = document.querySelector(".navbar-tab");
  var navbar = document.querySelector(".navbar-content");
  const elemWidth = tab.offsetWidth;

  if (tab.style.display === "flex") {
    navbar.style.marginLeft = elemWidth+'px';
  }
  else {
    navbar.style.marginLeft = '0px';
  }
}

function changeTab() {
  var tab = document.querySelector(".navbar-tab");
  var icon = document.getElementById("menuIcon");

  if (tab.style.display === "none") {
    tab.style.display = "flex";
    icon.classList.remove("fa-bars");
    icon.classList.add('fa-times');
  } 
  
  else {
    tab.style.display = "none";
    icon.classList.remove("fa-times");
    icon.classList.add("fa-bars");
  }
  marginChange();
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
