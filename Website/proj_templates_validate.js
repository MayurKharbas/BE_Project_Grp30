var attempt = 3;
function validate(){
var username = document.getElementById("u").value;
var password = document.getElementById("p").value;
if ( username == "admin" && password == "admin"){
alert ("Login successfully");
window.location.href = "www.google.com";
return false;
}
else{
attempt --;
alert("You have left "+attempt+" attempt;");
if( attempt == 0){
document.getElementById("u").disabled = true;
document.getElementById("p").disabled = true;
document.getElementById("s").disabled = true;
return false;
}
}
}