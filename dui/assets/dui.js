function buildCreationButton(){
    let createButtonDiv = document.createElement("div");
    createButtonDiv.className="col-auto";
    let createButton = document.createElement("button")
    createButton.className = "bi bi-plus rounded-circle btn btn-outline-primary";
    createButtonDiv.appendChild(createButton);
    return createButtonDiv;
}

function buildInputField(classname){
    let inputDiv = document.createElement("div");
    inputDiv.className= classname.replace(" form-control", "")+"_col"+" col";
    let inputField = document.createElement("input");
    inputField.className = classname;
    inputField.type = "text";
    inputField.step = "any";
    inputDiv.appendChild(inputField);
    return inputDiv;
}

function buildDeleteButton(){
    let deleteButtonDiv = document.createElement("div");
    deleteButtonDiv.className="col";
    let deleteButton = document.createElement("button")
    deleteButton.className = "bi bi-trash btn btn-outline-secondary";
    deleteButtonDiv.appendChild(deleteButton);
    return deleteButtonDiv;
}

function buildEditButton(){
    let editButtonDiv = document.createElement("div");
            editButtonDiv.className="col-auto";
            let editButton = document.createElement("button");
            editButton.className = "editbutton bi bi-pencil btn btn-outline-secondary";
            editButtonDiv.appendChild(editButton);

            return editButtonDiv;
}


function deleteParameterRow(button){
            if (button.parentNode.parentNode.className === "parent row"){
                let uniter = button.parentNode.parentNode.parentNode;
                uniter.remove();
            }

            if (button.parentNode.parentNode.className === "child row"){
                let uniter = button.parentNode.parentNode.parentNode;
                let child = button.parentNode.parentNode;

                child.remove();

                if (uniter.className === "uniter row"){
                    if (uniter.children.length===1){
                        uniter.children[0].className = "child row";
                        uniter.replaceWith(uniter.children[0])
                        // TODO: recover valuefield_col here
                    }
                }
            }
}

function onDeleteParameterRowClick(callback) {
        let deleteButtons = document.getElementsByClassName("bi-trash");
        let i;
        for (i = 0; i < deleteButtons.length; i++) {
            deleteButtons[i].onclick = function (button) {
                return function () { callback(button);};
            }(deleteButtons[i]);
        }
}

function onConfigRowClick(tableId, callback) {
        let table = document.getElementById(tableId);
        if (table == null){
            return null
            }

        let rows = table.getElementsByTagName("tr");
        let i;

        for (i = 2; i < rows.length; i++) {
            rows[i].onclick = function (row) {
                return function () {
                    callback(row);
                };
            }(rows[i]);
        }
}

function openConfig(row){
     location.href = location.protocol+"//"+location.host+"/config/"+row.textContent;
}

function onAddParameterRowClick(callback) {
        let addButtons = document.getElementsByClassName("bi-plus");
        let i;
        for (i = 0; i < addButtons.length; i++) {
            addButtons[i].onclick = function (button) {
                return function () {
                    callback(button);
                };
            }(addButtons[i]);
        }
}

function addParameterRow(button){
            let row = document.createElement("div");

            row.className="child row"
            let p = button.parentNode.parentNode.style.marginLeft; // return value in px; i.e 50px
            p = p.substring(0,p.length-2); // remove px ie : 50px becomes 50
            row.style.marginLeft = (+p)+ 100 +'px' // convert p to number and add 10

            //////////////////////////////////////////////////////////////////////////////////////////
            let modalEditor = document.createElement("div");
            modalEditor.className="modalEditor";
            row.appendChild(modalEditor);

            let modal_content = document.createElement("div");
            modal_content.className="modal-content";
            let modal_configuration_parameters = document.createElement("div");
            modal_configuration_parameters.className="modal_configuration_parameters";
            modalEditor.appendChild(modal_content);
            modal_content.appendChild(modal_configuration_parameters);



            let modalEndwagonRow = document.createElement("div");
            modalEndwagonRow.className="modalEndwagon row";
            modalEndwagonRow.style.marginLeft = "0px";
            let modalEndwagonCol = document.createElement("div");
            modalEndwagonCol.className="col-auto";
            let modalEndwagonAddButton = document.createElement("button");
            modalEndwagonAddButton.className="bi bi-plus rounded-circle modalEndwagon btn btn-outline-primary";
            modalEndwagonCol.appendChild(modalEndwagonAddButton);
            modalEndwagonRow.appendChild(modalEndwagonCol);
            modal_configuration_parameters.appendChild(modalEndwagonRow);

            let lastRow = document.createElement("div");
            lastRow.className = "row"
            lastRow.style.marginTop = "30px";

            let saveCol = document.createElement("div");
            saveCol.className="col-auto";
            // saveCol.style.width = "auto";
            let saveButton = document.createElement("button");
            saveButton.className="modal_save_btn me-1 btn btn-primary";
            saveCol.appendChild(saveButton);
            saveButton.textContent = "Save";
            let closeCol = document.createElement("div");
            closeCol.className="col-auto";
            let closeButton = document.createElement("button");
            closeButton.className="me-1 close btn btn-primary";
            closeButton.textContent = "Cancel";
            closeCol.appendChild(closeButton);

            lastRow.appendChild(saveCol);
            lastRow.appendChild(closeCol);
            modal_content.appendChild(lastRow);

            closeButton.onclick = function() {modalEditor.style.display = "none";}
            saveButton.onclick = function () {saveConfig(saveButton);}
            //////////////////////////////////////////////////////////////////////////////////////////

            let createButtonDiv = buildCreationButton();
            row.appendChild(createButtonDiv);

            let inputKeyDiv = buildInputField("keyfield form-control");
            inputKeyDiv.children[0].setAttribute("list", "parameters");
            row.appendChild(inputKeyDiv);

            let inputValueDiv = buildInputField("valuefield form-control");
            row.appendChild(inputValueDiv);

            let deleteButtonDiv = buildDeleteButton();
            row.appendChild(deleteButtonDiv)


            if (button.parentNode.parentNode.className === "parent row"){
                button.parentNode.parentNode.parentNode.appendChild(row);
            }

            if (button.parentNode.parentNode.className === "child row"){
                button.parentNode.parentNode.className = "parent row"
                let valuefield_col = button.parentNode.parentNode.getElementsByClassName("valuefield_col")[0];
                valuefield_col.remove();

                let uniterRow = document.createElement("div");
                uniterRow.className="uniter row";
                uniterRow.appendChild(button.parentNode.parentNode.cloneNode(true));

                uniterRow.appendChild(row);
                let parent = button.parentNode.parentNode;

                parent.replaceWith(uniterRow);
            }

            if (button.className.includes("endwagon")){
                let conf = document.getElementById("configuration_parameters");
                let endwagon = document.getElementById("endwagon")
                row.style.marginLeft = "0px";
                conf.insertBefore(row, endwagon);

            }

            if (button.className.includes("modalEndwagon")){
                let conf = button.parentNode.parentNode.parentNode;
                let modalEndwagon = button.parentNode.parentNode;
                row.style.marginLeft = "0px";
                conf.insertBefore(row, modalEndwagon);
            }

            set_callbacks();
}

function onSaveConfigButtonClick(callback) {
        let saveButtons = document.getElementsByClassName("save_btn");
        let modalSaveButtons = document.getElementsByClassName("modal_save_btn");

        let i;
        for (i = 0; i < saveButtons.length; i++) {
            saveButtons[i].onclick = function (button) {
                return function () {
                    callback(button);
                };
            }(saveButtons[i]);
        }
        for (i = 0; i < modalSaveButtons.length; i++) {
            modalSaveButtons[i].onclick = function (button) {
                return function () {
                    callback(button);
                };
            }(modalSaveButtons[i]);
        }
}

function parseHtmlToDict(html_array){
    let dict = {};
    let parent = null;
    let i;

     for(i = 0; i < html_array.length; i++){
         if (html_array[i].className==="uniter row"){
             let dict2 = parseHtmlToDict(Array.from(html_array[i].children));
             if (parent===null){
                 dict = {...dict, ...dict2};
             }
             else{
                 dict[parent].push(dict2)
             }
         }
         if (html_array[i].className==="parent row"){
             parent = html_array[i].getElementsByClassName("keyfield")[0].value;
             dict[parent] = [];
         }
         if (html_array[i].className==="child row"){
             let element = html_array[i].cloneNode(true);
             let modalWindow = element.getElementsByClassName("modalEditor");
             if(modalWindow.length>0){
                 modalWindow[0].remove();
             }

             let k = element.getElementsByClassName("keyfield")[0].value;
             let v = element.getElementsByClassName("valuefield")[0].value;

             if (parent != null){
                 let new_dict = {};
                 new_dict[k] = v;
                 dict[parent].push(new_dict);
             }
             else{
                dict[k] = v;
             }
         }
     }

     return dict;

}

function saveConfig(button){
    let configuration_parameters;
    let config_name;
    let rows;
    if (button.className.includes("modal")){
        configuration_parameters = button.parentNode.parentNode.parentNode.getElementsByClassName("modal_configuration_parameters")[0];
        config_name = button.parentNode.parentNode.parentNode.getElementsByClassName("modal_config_name")[0].textContent;
        rows = configuration_parameters.children;


        rows = Array.from(rows);
        rows = rows.slice(0, rows.length-1);
    }
    else{
        configuration_parameters = document.getElementById("configuration_parameters");
        config_name = "experiments/" + document.getElementById("config_name").value;
        rows = configuration_parameters.children;

        rows = Array.from(rows);
        rows = rows.slice(2, rows.length-4);
    }

    let redFlag = false;
    let inputFields = configuration_parameters.getElementsByTagName("input");
    let i = 0;
    while (i<inputFields.length){
        if (inputFields[i].value===""){
            inputFields[i].style.borderColor = "red";
            redFlag = true;
        }
        i++;
    }

    if (!redFlag){
        let dict = parseHtmlToDict(rows);

     fetch(location.protocol+"//"+location.host+'/save_config', {
    method: 'POST',
    headers: {'Accept': 'application/json', 'Content-Type': 'application/json'},
    body: JSON.stringify({ "html": dict, "config_name": config_name })})
   .then(response => response.json())
   .then(response => {
       console.log(JSON.stringify(response));
       // location.href = location.protocol+"//"+location.host;
       let confirmSaveModal = document.getElementById("confirm_save_modal");
       confirmSaveModal.style.display = "block";
       confirmSaveModal.getElementsByClassName("modal-body")[0].textContent = "Configuration " + config_name + " is saved";

   })
    }
}

function onKeyFieldChange(callback) {
        let keyFields = document.getElementsByClassName("keyfield");
        let i;

        for (i = 0; i < keyFields.length; i++) {
            keyFields[i].onchange = function (keyfield) {
                return function () {
                    callback(keyfield);
                };
            }(keyFields[i]);
        }
}

function addEditionButtonOnOverridePresence(keyfield){
    let row = keyfield.parentNode.parentNode;
    let content = keyfield.value;

    if(content.includes("override /")) {
        let rowElements = row.children;
        let isThereEditionKey = false;

        let i = 0;
        while (i < rowElements.length) {
            let clsName = rowElements[i].children[0].className;

           if(clsName.includes("bi-pencil")){
               isThereEditionKey = true;
           }
            i++;
        }

        if(isThereEditionKey===false){
            let index = rowElements.length-1;
            let editButtonDiv = buildEditButton();
            rowElements[index].parentNode.insertBefore(editButtonDiv, rowElements[index]);
        }

    }
    else{

        let rowElements = row.children;
        let i = 0;
        while (i < rowElements.length) {
            let clsName = rowElements[i].children[0].className;

           if(clsName.includes("bi-pencil")){
               rowElements[i].remove()
           }
            i++;
        }
    }

    set_callbacks();

}

function onEditButtonClick(callback) {
        let editButtons = document.getElementsByClassName("bi-pencil");
        let i;
        for (i = 0; i < editButtons.length; i++) {
            editButtons[i].onclick = function (button) {
                return function () {
                    callback(button);
                };
            }(editButtons[i]);
        }
}

function createModalContentFromConfigDict(config, level = 0){
    let rows = [];
    for (const [key, value] of Object.entries(config)) {

        let type = typeof value;

        if(type === "object"){
            let children =  createModalContentFromConfigDict(value, level+1);

            let uniter = document.createElement("div");
            uniter.className="uniter row"
            let parent = document.createElement("div");
            parent.className="parent row";
            parent.style.marginLeft = 30*level +'px' // convert p to number and add 10

            let createButtonDiv = buildCreationButton();

            let inputKeyDiv = buildInputField("keyfield form-control");
            let inputKeyInput = inputKeyDiv.getElementsByClassName("keyfield")[0];
            inputKeyInput.value = key;

            let deleteButtonDiv = buildDeleteButton();

            parent.appendChild(createButtonDiv);
            parent.appendChild(inputKeyDiv);
            parent.appendChild(deleteButtonDiv);

            uniter.appendChild(parent);

            let ind =0;
            while (ind<children.length){
                uniter.appendChild(children[ind]);
                ind++;
            }
            rows.push(uniter);
        }
        else
        {
            let row = document.createElement("div");
            row.className="child row"
            row.style.marginLeft = 30*level +'px' // convert p to number and add 10


            let createButtonDiv = buildCreationButton();
            row.appendChild(createButtonDiv)

            let inputKeyDiv = buildInputField("keyfield form-control");
            let inputKeyInput = inputKeyDiv.getElementsByClassName("keyfield")[0];
            inputKeyInput.value = key;
            row.appendChild(inputKeyDiv);

            let inputValueDiv = buildInputField("valuefield form-control");
            let inputValueInput = inputValueDiv.getElementsByClassName("valuefield")[0];
            inputValueInput.value = value;
            row.appendChild(inputValueDiv);

            let deleteButtonDiv = buildDeleteButton();
            row.appendChild(deleteButtonDiv)


            rows.push(row);
        }
    }
    return rows;
}


function openModalWindowOnEditButtonClick(button){
    let modal = button.parentNode.parentNode.children[0];
    let modal_content = modal.getElementsByClassName("modal-content")[0];
    let modal_configuration_parameters = modal.getElementsByClassName("modal_configuration_parameters")[0];
    if (modal_configuration_parameters.children.length!==1){
        modal.style.display = "block";
        return null;
    }

    let fileName = "";
    let i=0;
    while (i<button.parentNode.parentNode.children.length){
        if (button.parentNode.parentNode.children[i].children[0].className.includes("valuefield")){
            let k = button.parentNode.parentNode.children[i-1].children[0].value.replace("override /", "");
            let v = button.parentNode.parentNode.children[i].children[0].value;
            fileName = k+"/"+v;
        }
        i++;
    }

    fetch(location.protocol+"//"+location.host+'/get_config?config_name=' + fileName)
    .then(res => res.json())
    .then(out => {
        let configurations = out["configuration_parameters"];
        let rows = createModalContentFromConfigDict(configurations);
        let i =0;
        while (i<rows.length){
            modal.getElementsByClassName("modal_configuration_parameters")[0].prepend(rows[i]);
            i++;
        }


        let old_name = modal_content.getElementsByClassName("modal_config_name");
        if (old_name.length>0){
            old_name[0].remove();
        }

        let nameOfConfig = document.createElement("h3");
        nameOfConfig.className = "modal_config_name";
        const text = document.createTextNode(fileName+".yaml");
        nameOfConfig.appendChild(text);
        modal_content.insertBefore(nameOfConfig, modal_content.children[0])

        modal.style.display = "block";
        set_callbacks();
    })
    .catch(err => { throw err });

}


function modals(){

    let modalWindows = document.getElementsByClassName("modalEditor");

    let i=0;
    while(i<modalWindows.length){
        let mw = modalWindows[i]
        let span = mw.getElementsByClassName("close")[0];

        span.onclick = function() {mw.style.display = "none";}
        i++;
    }
}

function change_config_name_and_url(){
    let config_name = document.getElementById("config_name");
    config_name.value = "duplicate " + config_name.value;
    history.pushState(null, 'FeaturePoints Login', location.protocol+"//"+location.host+"/config/"+config_name.value);
}

function onInputFieldClick(){

        let inputFields = document.getElementsByTagName("input");
        let i;

        for (i = 0; i < inputFields.length; i++) {
            inputFields[i].onclick = function (inputField) {
                return function () {
                    if (inputField.style.borderColor==="red"){
                        inputField.style.borderColor = "#ced4da";
                    }
                };
            }(inputFields[i]);
        }

}

$(document).ready(function () {
    if (location.href === location.protocol+"//"+location.host+"/") {
       location.href = location.protocol+"//"+location.host+"/config_list";
    }
});

function set_callbacks(){
        onConfigRowClick("table", openConfig);
        onDeleteParameterRowClick(deleteParameterRow);
        onAddParameterRowClick(addParameterRow);
        onSaveConfigButtonClick(saveConfig);
        onKeyFieldChange(addEditionButtonOnOverridePresence);

        onEditButtonClick(openModalWindowOnEditButtonClick);
        onInputFieldClick();
        modals();


        let duplicate_btn = document.getElementById("duplicate_btn");
        duplicate_btn.onclick = function(){change_config_name_and_url()};

        let confirm_save_btn = document.getElementById("confirm_save_btn");
        confirm_save_btn.onclick = function (){
            document.getElementById("confirm_save_modal").style.display = "none";
            let modalEditors = document.getElementsByClassName("modalEditor");
            for(let i=0; i<modalEditors.length; i++){
                modalEditors[i].style.display = "none";
            }};
}


document.querySelector("body").onload = function() {

    setTimeout(function () {
        set_callbacks();
    }, 1000);
}