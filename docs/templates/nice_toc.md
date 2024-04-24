
   <script>
   window.addEventListener('DOMContentLoaded', function() {
        (function($) {
   //Function to make the index toctree collapsible
   $(function () {
       $('div.body .toctree-l1')
           .click(function(event){
               if (event.target.tagName.toLowerCase() != "a") {
                   if ($(this).children('ul').length > 0) {
                        $(this).attr('data-content',
                            (!$(this).children('ul').is(':hidden')) ? '\u25ba' : '\u25bc');
                       $(this).children('ul').toggle();
                   }
                   return true; //Makes links clickable
               }
           })
           .mousedown(function(event){ return false; }) //Firefox highlighting fix
           .children('ul').hide();
       // Initialize the values
       $('div.body li.toctree-l1:not(:has(ul))').attr('data-content', '-');
       $('div.body li.toctree-l1:has(ul)').attr('data-content', '\u25ba');
       $('div.body li.toctree-l1:has(ul)').css('cursor', 'pointer');
   });
        })(jQuery);
    });
   </script>

  <style type="text/css">
    div.body li, div.body ul {
        transition-duration: 0.2s;
    }

  div.body li, div.body ul {
      transition-duration: 0.2s;
  }



  div.body li.toctree-l1 {
      padding: 0.25em 0 0.25em 0 ;
      list-style-type: none;
      background-color: #FFFFFF;
      font-size: 85% ;
      font-weight: normal;
      margin-left: 0;
  }

  div.body li.toctree-l1 ul {
      padding-left: 100px ;
  }

  div.body li.toctree-l1:before {
      content: attr(data-content);
      font-size: 1rem;
      color: #777;
      display: inline-block;
      width: 1.5rem;
  }

  div.body li.toctree-l3 {
      font-size: 88% ;
      list-style-type: square;
      font-weight: normal;
      margin-left: 0;
  }

  div.body li.toctree-l4 {
      font-size: 93% ;
      list-style-type: circle;
      font-weight: normal;
      margin-left: 0;
  }

  div.body div.topic li.toctree-l1 {
      font-size: 100% ;
      font-weight: bold;
      background-color: transparent;
      margin-bottom: 0;
      margin-left: 1.5em;
      display:inline;
  }

  div.body div.topic p {
      font-size: 90% ;
      margin: 0.4ex;
  }

  div.body div.topic p.topic-title {
      display:inline;
      font-size: 100% ;
      margin-bottom: 0;
  }

  </style>

   <style type="text/css">
    div.body div.toctree-wrapper ul {
        padding-left: 0;
    }


    div.body li.toctree-l1 {
        padding: 0 1em 0.5em 1em;
        list-style-type: none;
        font-size: 130%;
        font-weight: bold;
    }

    div.body li.toctree-l2 {
        font-size: 70%;
        list-style-type: square;
        font-weight: normal;
        margin-left: 40px;
    }

    div.body li.toctree-l3 {
        font-size: 85%;
        list-style-type: circle;
        font-weight: normal;
        margin-left: 40px;
    }

    div.body li.toctree-l4 {
        margin-left: 40px;
    }
  </style>
