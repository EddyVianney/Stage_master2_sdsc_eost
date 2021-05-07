/**************************
 Script to add the bootstrap
 form classes to the eprints
 workflow
***************************/
var j = jQuery.noConflict();

j(document).ready(function () {
	// used to flag required fields
        //var star='<span class="glyphicon glyphicon-star-empty"></span>';
        var star='<span class="required-star" style="color:red;">*</span>';

	// textarea stylings
	j('.ep_sr_component textarea').addClass('form-control');	
	
	//name entry style stuff (work in progress...)
	//j('.ep_form_input_grid input.ep_form_text').addClass('form-control');	
	
	//Select. Using the jquery selector here to only select dvisions as i'm worried about styling ALL selects in eprints
	j('select[id$=divisions]').addClass('form-control');	

	//subjects - stop page jump
	j('.ep_subjectinput_add_button, .ep_subjectinput_remove_button').click( function(){
		var $sub = j(this);
		if( comp = $sub.attr( "name" ).match( /_internal_([^_]+)_/ ) )
		{
			$form = $sub.parents( 'form' );
			act = $form.attr( "action" );
			$form.attr( "action", act.replace( "#t", "#"+comp[1] ) );
		}
	});
	
	// bootstrap radiobuttons
	j('input:radio').parent().parent().addClass("radio")

	//j('select[id$=divisions]').addClass('form-control');	
	//j('.ep_form_input_grid select').addClass('form-control');	
	j('.ep_form_input_grid select:not([name$="date_month"],[name$="date_day"],[name$="date_embargo_month"],[name$="date_embargo_day"])').addClass('form-control');
	
	//j('.ep_form_field_input .ep_sr_component').addClass('panel panel-default').removeClass('ep_sr_compenent');

	// Style the collapses bars to look okay
	j('.ep_sr_component .ep_sr_collapse_bar').addClass('panel-heading').removeClass('ep_sr_collapse_bar');




	//import form
	j('form table td:contains("Import from")').closest("form").addClass('form-inline import-form');
	j('.import-form select').addClass('form-control').wrap('<div class="form-group"></div>');

	
	//add column form
	j('.ep_columns_add form').addClass('form-inline').children().wrap('<div class="form-group"></div>');
	j('.ep_columns_add form select').addClass('form-control');

	//simple search form
	j('main form[action="/cgi/search/simple"]').addClass('form-inline search-form');
	j('.search-form input[name="q"]').addClass('form-control').attr('placeholder','Search the repository').wrap('<div class="form-group"></div>');

	//more of the workflow bootstrapped		
	//j('table.ep_form_input_grid input[type="text"]').addClass('form-control');
	j('table.ep_form_input_grid input[type="text"]:not([name$="date_year"],[name$="date_embargo_year"])').addClass('form-control');
	j("[id$='pagerange_to']").parent().addClass('form-inline')
	
	j(".ep_form_field_input")
        .observe('childlist', ".epjs_ajax", function(record) {
              console.log("mut:"+record);
                updateButtons();
		j(this).closest('table').addClass('ajaxedTable');
		j('table.ajaxedTable tr td input:not(.form-control)').not('.ep_form_internal_button').addClass('form-control');
        });

	// JLRS
	//
	// This doesn't work for ajax-loaded checkboxes.
	j('label:has(input[type="checkbox"])').css('font-weight','normal');


/////// NEXT TWO ROWS! CHANGE TO 'find reason input row'
	j("body").on( "change", 'input[name$=embargo_indefinitely]', function(){ 
                if(this.checked){
			flag_embargo_reason(j(this).parents('div.ep_upload_doc'), true);
		} else {
			flag_embargo_reason(j(this).parents('div.ep_upload_doc'), false);
		}
	});
	j("body").on( "change", 'select[id$=embargo_period]', function(){
                if(this.value > 0){
			flag_embargo_reason(j(this).parents('div.ep_upload_doc'), true);
                } else {
			flag_embargo_reason(j(this).parents('div.ep_upload_doc'), false);
                };
        });
	j("body").on( "change", 'input[id$=_date_embargo_year]', function(){
                if(this.value != ''){
			flag_embargo_reason(j(this).parents('div.ep_upload_doc'), true);
                } else {
			flag_embargo_reason(j(this).parents('div.ep_upload_doc'), false);
                };
        });
	
	// make default display correct
	j('div.ep_upload_doc').each(function(){
		flag_embargo_reason(j(this), false);
	});

	j("body").on( "change",'input[type=radio][name$=embargo_reason]', function(){
		// use DOM traveral to stay in the same doc
                if(this.value == 'freetext' ){
                        j(this).parents('tr').next('tr').find('textarea[id$=embargo_reason_freetext]').parents('tr').find('th').prepend(star);
                } else {
                        j(this).parents('tr').next('tr').find('textarea[id$=embargo_reason_freetext]').parents('tr').find('th.ep_multi_heading span.required-star').remove();
                };
        });

	j('input[type=radio][name$=embargo_reason][value=freetext]').each(function(){
		if( j(this).prop( "checked" ) ){
			j(this).parents('tr').find('th').prepend(star);
		}
	});

	// hids day input (and label) for date field
	j('select[id$=_date_day]').filter(function(){ return this.id.match(/^c\d+_date_day$/) }).hide().parent().contents().filter(function () {
	        return this.nodeType === Node.TEXT_NODE;
	    }).last().remove();

	j('.ep_upload_doc_content select[name$="_content"]').each(function(){
		if( !j(this).val() ){
			j(this).prepend('<option selected="true" disabled="disabled">Please choose...</option>');
		}
	});
	//j('.ep_upload_doc_content select[name$="_embargo_period"]').each(function(){
	//	if( !j(this).selected ){
	//		j(this).prepend('<option selected="true" disabled="disabled">Please choose...</option>');
	//	}
	//});


        // If there is only one document, hide the re-order buttons
        if( j('.ep_form_field_input div.ep_upload_doc').length == 1 ){
        	j('a[name$="EPrint::Document::MoveDown"],a[name$="EPrint::Document::MoveUp"]').hide();
        	j('input[name$="EPrint::Document::MoveDown"],input[name$="EPrint::Document::MoveUp"]').hide();
        }

	
	//j("body").on( "change",'.ep_form_field_input:has(.ep_upload_doc)', function(){
	//j("body").on( "change", function(){
	//j(document).on( "change", function(){
	j("body").on( "DOMSubtreeModified",'.ep_form_field_input:has(.ep_upload_doc)', function(){
        	if( j('.ep_form_field_input div.ep_upload_doc').length == 1 ){
        		j('a[name$="EPrint::Document::MoveDown"],a[name$="EPrint::Document::MoveUp"]').hide();
        		j('input[name$="EPrint::Document::MoveDown"],input[name$="EPrint::Document::MoveUp"]').hide();
        	} else {
        		j('a[name$="EPrint::Document::MoveDown"],a[name$="EPrint::Document::MoveUp"]').show();
        		j('input[name$="EPrint::Document::MoveDown"],input[name$="EPrint::Document::MoveUp"]').show();

		}
	});
	
	j("body").on( "DOMSubtreeModified",'#ep_messages', function(){

	});

function flag_embargo_reason($doc, required){
	var $indef = $doc.find('input[name$=_embargo_indefinitely]');
	var $emb_date = $doc.find('input[name$=_date_embargo_year]');
	var $emb_period = $doc.find('select[name$=_embargo_period]');

	if( 
		( $indef.length > 0 && $indef.is(':checked') ) || 
		( $emb_date.length > 0 && $emb_date.val() !='' ) || 
		( $emb_period.length > 0 && $emb_period.val() > 0 ) 
	){
		//if one of the inputs that makes a resaon needed is 'true'
		required = true;
	}
	
	var $th = $doc.find('tr:has(input[id$=_embargo_reason_]) th');
	var $td = $doc.find('td:has(input[id$=_embargo_reason_])');
	if( required ){
		if( $td.has('.wreo_embargo').length == 0 ){
			$td.prepend('<p class="wreo_embargo"><strong>Please select the reason for applying an embargo:</strong></p>');
		}
		$td.find('div:has(label[for$=_embargo_reason_])').hide();
		if( $th.has('span.required-star').length ==0 ){
			$th.prepend(star);
		}
	} else {
		$th.find('span.required-star').remove();
		$td.find('.wreo_embargo').remove();
		$td.find('div:has(label[for$=_embargo_reason_])').show();
	}
} //end flag_embargo_reason

});
