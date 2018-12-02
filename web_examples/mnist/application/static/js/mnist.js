var mouse_status=0; // up

$(document).ready(function(){

    $("#clear").click(function(){
        mouse_status = 0;
        $("#result").text("")
        $(".draw_cell").css("background-color", "#bddfff");
        $(".value_holder").val(0);
    });

    $(".draw_cell").mouseenter(function(event){
        if(mouse_status == 1){
            event.target.style.backgroundColor = "red";
            id = "h_" + event.target.id;
            $("#"+id).val(1);
        }
    });

    $(".draw_cell").mousedown(function(){
        mouse_status = 1; // down
    });

    $(".draw_cell").mouseup(function(){
        mouse_status = 0; // up
    });

    $(function() {
    $('#predict').bind('click', function() {
        mouse_status = 0;
        $("#result").text("")

        var data_to_send = $(".value_holder").serialize();

        $.getJSON('/api/predict', data_to_send, function(result) {
            $("#result").text(result);
      });
      return false;
    });
  });
});
