(self.webpackChunkploomber_extension=self.webpackChunkploomber_extension||[]).push([[104],{4104:(e,t,n)=>{"use strict";n.r(t),n.d(t,{MODULE_NAME:()=>E,MODULE_VERSION:()=>w,default:()=>C});var o=n(1708),l=n(2292);const i="ploomber-extension:settings",r=new l.Signal({}),s={id:i,autoStart:!0,requires:[o.ISettingRegistry],activate:(e,t,n)=>{function o(e){const t=e.get("showShareNotebook").composite;r.emit({showShareNotebook:t})}Promise.all([e.restored,t.load(i)]).then((([,e])=>{o(e),e.changed.connect(o)})).catch((e=>{console.error(`Something went wrong when reading the settings.\n${e}`)}))}};var a=n(3908),c=n(2508),p=n(6512),d=n.n(p),u=n(5176),m=n(5124);var b=n(3440),y=n(9008);async function h(e="",t={}){const n=y.ServerConnection.makeSettings(),o=b.URLExt.join(n.baseUrl,"ploomber",e);let l;try{l=await y.ServerConnection.makeRequest(o,t,n)}catch(e){throw new y.ServerConnection.NetworkError(e)}let i=await l.text();if(i.length>0)try{i=JSON.parse(i)}catch(e){console.log("Not a JSON response body.",l)}if(!l.ok)throw new y.ServerConnection.ResponseError(l,i.message||i);return i}const g=e=>{var t,n,o,l,i,r;return"missing file"==(null===(t=null==e?void 0:e.message)||void 0===t?void 0:t.type)?d().createElement("div",{"data-testid":"error-message-area"},d().createElement(u.Typography,{variant:"subtitle1",gutterBottom:!0}," A ",d().createElement("code",null,null===(o=null===(n=null==e?void 0:e.message)||void 0===n?void 0:n.detail)||void 0===o?void 0:o.fileName)," file with dependencies is required to deploy your notebook. Please add it at ",d().createElement("code",null,null===(i=null===(l=null==e?void 0:e.message)||void 0===l?void 0:l.detail)||void 0===i?void 0:i.filePath),". To learn more, see the ",d().createElement("a",{target:"_blank",rel:"noopener noreferrer",href:"https://docs.cloud.ploomber.io/en/latest/apps/jupyterlab-plugin.html"},"docs"))):d().createElement(u.Typography,{variant:"subtitle1",gutterBottom:!0},null===(r=null==e?void 0:e.message)||void 0===r?void 0:r.detail)},f=e=>{const t=e.notebook_path,[n,o]=(0,p.useState)(!0),[l,i]=(0,p.useState)(!1),[r,s]=(0,p.useState)(!1),[a,c]=(0,p.useState)(!1),[b,y]=(0,p.useState)(""),[f,v]=(0,p.useState)(""),[x,j]=(0,p.useState)("init"),[k,w]=(0,p.useState)(null),[E,C]=(0,p.useState)("");(0,p.useEffect)((()=>{S()}),[]),(0,p.useEffect)((()=>{"success"===x&&c(!0)}),[x]);const S=async()=>{await h("apikey").then((e=>{null!=(null==e?void 0:e.data)&&(y(e.data),j("success"))})).catch((e=>{console.error(`The ploomber_extension server extension appears to be missing.\n${e}`)})),o(!1)},N={init:{label:"API Key",variant:"outlined",color:"primary"},success:{label:"Valid API Key",variant:"filled",color:"success"},error:{label:"Please enter valid API Key",variant:"filled",color:"warning"}};return d().createElement(u.Box,{p:6,style:{width:600}},n||l?d().createElement(u.Box,{sx:{display:"flex",justifyContent:"center",alignItems:"center"}},d().createElement(u.CircularProgress,null)):d().createElement(d().Fragment,null,d().createElement(u.Grid,{container:!0,spacing:4,alignItems:"center",direction:"column"},"success"!==x&&d().createElement("div",null,d().createElement(u.Grid,{item:!0,container:!0,direction:"row",alignItems:"center",justifyContent:"flex-start",width:"100%",my:2},"Upload this notebook to Ploomber Cloud to share it with anyone. ",d().createElement(u.Link,{href:"https://docs.cloud.ploomber.io/en/latest/examples/voila.html",target:"_blank",rel:"noopener noreferrer"}," Click here to learn more.")),d().createElement(u.Grid,{item:!0,container:!0,direction:"row",alignItems:"center",width:"100%"},d().createElement(u.Grid,{container:!0,direction:"row",alignItems:"center",spacing:1},d().createElement(u.Grid,{item:!0,xs:10},d().createElement(u.TextField,{id:"api-key-input",size:"small",onChange:e=>{y(e.target.value)},value:b,label:N[x].label,variant:N[x].variant,color:N[x].color,error:"error"==x,fullWidth:!0,focused:!0})),d().createElement(u.Grid,{item:!0,xs:2,alignItems:"center",justifyContent:"center"},d().createElement(u.Button,{onClick:async()=>{o(!0);const e={api_key:b};await h("apikey",{body:JSON.stringify(e),method:"POST"}).then((e=>{"success"==(null==e?void 0:e.result)?j("success"):j("error")})).catch((t=>{console.error(`Error on POST ${e}.\n${t}`)})),o(!1)},variant:"contained",size:"small"},"CONFIRM")))),d().createElement(u.Grid,{item:!0,container:!0,direction:"row",alignItems:"center",width:"100%",my:2},"You need an API key to upload this notebook. ",d().createElement(u.Link,{href:"https://docs.cloud.ploomber.io/en/latest/quickstart/apikey.html",target:"_blank",rel:"noopener noreferrer"},"Click here to get an API Key"))),"success"===x&&d().createElement(u.Grid,{item:!0,container:!0,alignItems:"center",spacing:4,direction:"column"},a?d().createElement(d().Fragment,null,d().createElement(d().Fragment,null,d().createElement(u.Typography,{variant:"subtitle1",gutterBottom:!0},"Confirm that you want to upload this notebook to Ploomber Cloud"),d().createElement(u.Button,{onClick:async()=>{c(!1),await(async()=>{i(!0);const n={notebook_path:t,api_key:b};await h("nb-upload",{body:JSON.stringify(n),method:"POST"}).then((t=>{var n,o=t.deployment_result,l={type:"generic",detail:""};"missing file"===(null==o?void 0:o.type)&&(null==o?void 0:o.detail)?(l.type=o.type,l.detail={fileName:"requirements.txt",filePath:o.detail},w(l)):(null==o?void 0:o.detail)||(null==o?void 0:o.message)?(l.detail=o.detail||o.message,w(l)):(v("https://www.platform.ploomber.io/notebooks/"+(null==o?void 0:o.id)),e.metadata.ploomber={project_id:null==o?void 0:o.project_id},null===(n=null==e?void 0:e.context)||void 0===n||n.save())})),i(!1)})()},variant:"contained",size:"small",color:"primary",disabled:""!==f,endIcon:d().createElement(m.c,null)},"CONFIRM "))):d().createElement(d().Fragment,null,k?d().createElement(g,{message:k}):d().createElement(d().Fragment,null,d().createElement(u.Grid,{item:!0,justifyContent:"center",xs:12},"Your notebook is available here:"),d().createElement(u.Grid,{item:!0,justifyContent:"center",xs:12},d().createElement(u.Chip,{label:f,variant:"outlined",onClick:()=>{window.open(f),s(!0),C("Deployment Success")}}),d().createElement(u.Snackbar,{open:r,onClose:()=>s(!1),autoHideDuration:2e3,message:E}))))))))};a.ReactWidget;class v extends a.ReactWidget{constructor(e){super(),this.state={notebookPath:e.notebookPath,metadata:e.metadata,context:e.context}}render(){return d().createElement(f,{notebook_path:this.state.notebookPath,metadata:this.state.metadata,context:this.state.context})}}class x{constructor(){this._onSettingsChanged=(e,t)=>{t.showShareNotebook?this.panel.toolbar.insertItem(10,"deployNB",this.deployNotebookButton):this.deployNotebookButton.parent=null},r.connect(this._onSettingsChanged)}createNew(e,t){return this.panel=e,this.deployNotebookButton=new a.ToolbarButton({className:"share-nb-button",label:"Share Notebook",onClick:()=>{!function(e,t){const n=new v({notebookPath:e.context.contentsModel.path,metadata:e.model.metadata,context:t});new a.Dialog({title:"Share Notebook",body:n,buttons:[{ariaLabel:"Close dialog",label:"Close",caption:"",className:"bg-info",accept:!1,actions:[],displayType:"default",iconClass:"",iconLabel:""}]}).launch()}(e,t)},tooltip:"Share notebook by uploading it to Ploomber Cloud"}),this.deployNotebookButton.node.setAttribute("data-testid","share-btn"),e.toolbar.insertItem(10,"deployNB",this.deployNotebookButton),new c.DisposableDelegate((()=>{this.deployNotebookButton.dispose()}))}}const j={activate:e=>{e.docRegistry.addWidgetExtension("Notebook",new x)},autoStart:!0,id:"sharing",requires:[]},k=n(6604),w=k.version,E=k.name,C=[s,j]},492:e=>{e.exports=function(e){return e&&e.__esModule?e:{default:e}},e.exports.__esModule=!0,e.exports.default=e.exports},5124:(e,t,n)=>{"use strict";var o=n(492);t.c=void 0;var l=o(n(3540)),i=n(7624);t.c=(0,l.default)((0,i.jsx)("path",{d:"M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96M19 18H6c-2.21 0-4-1.79-4-4s1.79-4 4-4h.71C7.37 7.69 9.48 6 12 6c3.04 0 5.5 2.46 5.5 5.5v.5H19c1.66 0 3 1.34 3 3s-1.34 3-3 3"}),"CloudQueue")},3540:(e,t,n)=>{"use strict";Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"default",{enumerable:!0,get:function(){return o.createSvgIcon}});var o=n(5336)},6604:e=>{"use strict";e.exports=JSON.parse('{"name":"ploomber-extension","version":"0.1.0","description":"A JupyterLab extension.","keywords":["jupyter","jupyterlab","jupyterlab-extension"],"homepage":"https://github.com/ploomber/ploomber-extension.git","bugs":{"url":"https://github.com/ploomber/ploomber-extension.git/issues"},"license":"BSD-3-Clause","author":{"name":"Ploomber","email":"contact@ploomber.io"},"files":["lib/**/*.{d.ts,eot,gif,html,jpg,js,js.map,json,png,svg,woff2,ttf}","style/**/*.{css,js,eot,gif,html,jpg,json,png,svg,woff2,ttf}","src/**/*.{ts,tsx}","settings-schema/**/*.json"],"main":"lib/index.js","types":"lib/index.d.ts","style":"style/index.css","repository":{"type":"git","url":"https://github.com/ploomber/ploomber-extension.git.git"},"scripts":{"build":"jlpm build:lib && jlpm build:labextension:dev","build:prod":"jlpm clean && jlpm build:lib:prod && jlpm build:labextension","build:labextension":"jupyter labextension build .","build:labextension:dev":"jupyter labextension build --development True .","build:lib":"tsc --sourceMap","build:lib:prod":"tsc","clean":"jlpm clean:lib","clean:lib":"rimraf lib tsconfig.tsbuildinfo","clean:lintcache":"rimraf .eslintcache .stylelintcache","clean:labextension":"rimraf ploomber_extension/labextension ploomber_extension/_version.py","clean:all":"jlpm clean:lib && jlpm clean:labextension && jlpm clean:lintcache","eslint":"jlpm eslint:check --fix","eslint:check":"eslint src/ --cache --ext .ts,.tsx","install:extension":"jlpm build","lint":"jlpm stylelint && jlpm prettier && jlpm eslint","lint:check":"jlpm stylelint:check && jlpm prettier:check && jlpm eslint:check","prettier":"jlpm prettier:base --write --list-different","prettier:base":"prettier \\"**/*{.ts,.tsx,.js,.jsx,.css,.json,.md}\\"","prettier:check":"jlpm prettier:base --check","stylelint":"jlpm stylelint:check --fix","stylelint:check":"stylelint --cache \\"style/**/*.css\\"","test":"jest --coverage","watch":"run-p watch:src watch:labextension","watch:src":"tsc -w --sourceMap","watch:labextension":"jupyter labextension watch ."},"dependencies":{"@emotion/react":"^11.11.3","@emotion/styled":"^11.11.0","@jupyterlab/application":"^4.0.0","@jupyterlab/coreutils":"^6.0.0","@jupyterlab/services":"^7.0.0","@mui/icons-material":"^5.15.7","@mui/material":"^5.15.7","@testing-library/jest-dom":"^6.4.1","@testing-library/react":"^14.2.1","@types/jest-when":"^3.5.5"},"devDependencies":{"@jupyterlab/builder":"^4.0.0","@jupyterlab/testutils":"^4.0.0","@playwright/test":"^1.41.2","@types/jest":"^29.2.0","@types/jest-when":"^3.5.5","@types/json-schema":"^7.0.11","@types/node":"^20.11.16","@types/react":"^18.0.26","@types/react-addons-linked-state-mixin":"^0.14.22","@typescript-eslint/eslint-plugin":"^6.1.0","@typescript-eslint/parser":"^6.1.0","css-loader":"^6.7.1","eslint":"^8.36.0","eslint-config-prettier":"^8.8.0","eslint-plugin-prettier":"^5.0.0","jest":"^29.2.0","jest-when":"^3.5.2","mkdirp":"^1.0.3","npm-run-all":"^4.1.5","prettier":"^3.0.0","rimraf":"^5.0.1","source-map-loader":"^1.0.2","style-loader":"^3.3.1","stylelint":"^15.10.1","stylelint-config-recommended":"^13.0.0","stylelint-config-standard":"^34.0.0","stylelint-csstree-validator":"^3.0.0","stylelint-prettier":"^4.0.0","typescript":"~5.0.2","yjs":"^13.5.0"},"sideEffects":["style/*.css","style/index.js"],"styleModule":"style/index.js","publishConfig":{"access":"public"},"jupyterlab":{"extension":true,"schemaDir":"settings-schema","outputDir":"ploomber_extension/labextension","sharedPackages":{}},"eslintConfig":{"extends":["eslint:recommended","plugin:@typescript-eslint/eslint-recommended","plugin:@typescript-eslint/recommended","plugin:prettier/recommended"],"parser":"@typescript-eslint/parser","parserOptions":{"project":"tsconfig.json","sourceType":"module"},"plugins":["@typescript-eslint"],"rules":{"@typescript-eslint/naming-convention":["error",{"selector":"interface","format":["PascalCase"],"custom":{"regex":"^I[A-Z]","match":true}}],"@typescript-eslint/no-unused-vars":["warn",{"args":"none"}],"@typescript-eslint/no-explicit-any":"off","@typescript-eslint/no-namespace":"off","@typescript-eslint/no-use-before-define":"off","@typescript-eslint/quotes":["error","single",{"avoidEscape":true,"allowTemplateLiterals":false}],"curly":["error","all"],"eqeqeq":"error","prefer-arrow-callback":"error"}},"prettier":{"singleQuote":true,"trailingComma":"none","arrowParens":"avoid","endOfLine":"auto","overrides":[{"files":"package.json","options":{"tabWidth":4}}]},"stylelint":{"extends":["stylelint-config-recommended","stylelint-config-standard","stylelint-prettier/recommended"],"plugins":["stylelint-csstree-validator"],"rules":{"csstree/validator":true,"property-no-vendor-prefix":null,"selector-class-pattern":"^([a-z][A-z\\\\d]*)(-[A-z\\\\d]+)*$","selector-no-vendor-prefix":null,"value-no-vendor-prefix":null}}}')}}]);