/**************************
+ Script to add make the 
+ eprints workflow more 
+ bootstrappy 
+***************************/

var j = jQuery.noConflict();

j(document).ready(function () {

	//Make panels in workflow stages
        j('.ep_form_field_input .ep_sr_component').addClass('panel panel-default').removeClass('ep_sr_compenent');
        j('.ep_form_field_input .ep_sr_title_bar').addClass('panel-heading').removeClass('ep_sr_title_bar');
        j('.ep_form_field_input .ep_sr_content').addClass('panel-body').removeClass('ep_sr_content');
        j('.panel table.table th').css('border','none');
        j('.panel table.table td.ep_form_input_grid_arrows').css('border-bottom','1px solid #ddd');

	//Make whole thing a row and add cols
	j("main > div > form[action='/cgi/users/home#t']").addClass('row');
	j('.ep_form_field_input').removeClass("col-md-5").addClass('col-md-12');
	//j("main > div > form[action='/cgi/users/home#t'] textarea, main > div > form[action='/cgi/users/home#t'] select, main > div > form[action='/cgi/users/home#t'] table.ep_form_input_grid input").addClass('form-control');
	//j("main > div > form[action='/cgi/users/home#t'] table.ep_form_input_grid input").css("margin","2px");
	
	// fix EPrints collapsed elements title bars
	j('div[id^=c][id$=_col].ep_toggle').addClass( 'ep_sr_title' );

	// If there is only one document, hide the re-order buttons
	//if( j('.ep_form_field_input div.ep_upload_doc').length == 1 ){
	//	j('a[name$="EPrint::Document::MoveDown"],a[name$="EPrint::Document::MoveUp"]').hide();
	//}
	
	// Make text areas wider
	j('.ep_form_input_grid:has(textarea)').css( 'width', '90%' );
});
