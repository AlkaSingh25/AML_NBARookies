var _JUPYTERLAB;(()=>{"use strict";var e,r,t,a,o,n,i,l,f,u,d,s,c,b,p,h,m,v,g,y,w,P,S={2608:(e,r,t)=>{var a={"./index":()=>Promise.all([t.e(336),t.e(512),t.e(775),t.e(104)]).then((()=>()=>t(4104))),"./extension":()=>Promise.all([t.e(336),t.e(512),t.e(775),t.e(104)]).then((()=>()=>t(4104))),"./style":()=>t.e(736).then((()=>()=>t(8736)))},o=(e,r)=>(t.R=r,r=t.o(a,e)?a[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),n=(e,r)=>{if(t.S){var a="default",o=t.S[a];if(o&&o!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[a]=e,t.I(a,r)}};t.d(r,{get:()=>o,init:()=>n})}},x={};function j(e){var r=x[e];if(void 0!==r)return r.exports;var t=x[e]={id:e,exports:{}};return S[e](t,t.exports,j),t.exports}j.m=S,j.c=x,j.n=e=>{var r=e&&e.__esModule?()=>e.default:()=>e;return j.d(r,{a:r}),r},j.d=(e,r)=>{for(var t in r)j.o(r,t)&&!j.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},j.f={},j.e=e=>Promise.all(Object.keys(j.f).reduce(((r,t)=>(j.f[t](e,r),r)),[])),j.u=e=>e+"."+{28:"1f003d1611e9a92b831a",104:"447122088b5cb74ef45d",132:"a11bf2d79b83c55b689a",156:"491a93129c84d26e6681",204:"89d5b0c27f8042a09438",336:"e55f4171dbb9a2836412",408:"f1c11b0d4cc82551e2d1",448:"858dd0074eeca58d9b4b",512:"2e0d38cf0f890e823b4d",584:"f84452a33977ae03c5cb",736:"51427e9d0a00dd2006b4",775:"871c8c2402bd7b41eed2",816:"467885bd76dcbbe1351f",912:"ca8a6462b39b5e1f69ba",944:"bb70927aee29873b11e2"}[e]+".js?v="+{28:"1f003d1611e9a92b831a",104:"447122088b5cb74ef45d",132:"a11bf2d79b83c55b689a",156:"491a93129c84d26e6681",204:"89d5b0c27f8042a09438",336:"e55f4171dbb9a2836412",408:"f1c11b0d4cc82551e2d1",448:"858dd0074eeca58d9b4b",512:"2e0d38cf0f890e823b4d",584:"f84452a33977ae03c5cb",736:"51427e9d0a00dd2006b4",775:"871c8c2402bd7b41eed2",816:"467885bd76dcbbe1351f",912:"ca8a6462b39b5e1f69ba",944:"bb70927aee29873b11e2"}[e],j.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),j.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="ploomber-extension:",j.l=(t,a,o,n)=>{if(e[t])e[t].push(a);else{var i,l;if(void 0!==o)for(var f=document.getElementsByTagName("script"),u=0;u<f.length;u++){var d=f[u];if(d.getAttribute("src")==t||d.getAttribute("data-webpack")==r+o){i=d;break}}i||(l=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,j.nc&&i.setAttribute("nonce",j.nc),i.setAttribute("data-webpack",r+o),i.src=t),e[t]=[a];var s=(r,a)=>{i.onerror=i.onload=null,clearTimeout(c);var o=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),o&&o.forEach((e=>e(a))),r)return r(a)},c=setTimeout(s.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=s.bind(null,i.onerror),i.onload=s.bind(null,i.onload),l&&document.head.appendChild(i)}},j.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},(()=>{j.S={};var e={},r={};j.I=(t,a)=>{a||(a=[]);var o=r[t];if(o||(o=r[t]={}),!(a.indexOf(o)>=0)){if(a.push(o),e[t])return e[t];j.o(j.S,t)||(j.S[t]={});var n=j.S[t],i="ploomber-extension",l=(e,r,t,a)=>{var o=n[e]=n[e]||{},l=o[r];(!l||!l.loaded&&(!a!=!l.eager?a:i>l.from))&&(o[r]={get:t,from:i,eager:!!a})},f=[];return"default"===t&&(l("@emotion/react","11.11.3",(()=>Promise.all([j.e(944),j.e(28),j.e(512),j.e(204)]).then((()=>()=>j(1028))))),l("@emotion/styled","11.11.0",(()=>Promise.all([j.e(156),j.e(512),j.e(912),j.e(816),j.e(408)]).then((()=>()=>j(156))))),l("@mui/material","5.15.7",(()=>Promise.all([j.e(944),j.e(448),j.e(336),j.e(512),j.e(912),j.e(775),j.e(584)]).then((()=>()=>j(9448))))),l("ploomber-extension","0.1.0",(()=>Promise.all([j.e(336),j.e(512),j.e(775),j.e(104)]).then((()=>()=>j(4104)))))),e[t]=f.length?Promise.all(f).then((()=>e[t]=1)):1}}})(),(()=>{var e;j.g.importScripts&&(e=j.g.location+"");var r=j.g.document;if(!e&&r&&(r.currentScript&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");if(t.length)for(var a=t.length-1;a>-1&&!e;)e=t[a--].src}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),j.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),a=t[1]?r(t[1]):[];return t[2]&&(a.length++,a.push.apply(a,r(t[2]))),t[3]&&(a.push([]),a.push.apply(a,r(t[3]))),a},a=(e,r)=>{e=t(e),r=t(r);for(var a=0;;){if(a>=e.length)return a<r.length&&"u"!=(typeof r[a])[0];var o=e[a],n=(typeof o)[0];if(a>=r.length)return"u"==n;var i=r[a],l=(typeof i)[0];if(n!=l)return"o"==n&&"n"==l||"s"==l||"u"==n;if("o"!=n&&"u"!=n&&o!=i)return o<i;a++}},o=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var a=1,n=1;n<e.length;n++)a--,t+="u"==(typeof(l=e[n]))[0]?"-":(a>0?".":"")+(a=2,l);return t}var i=[];for(n=1;n<e.length;n++){var l=e[n];i.push(0===l?"not("+f()+")":1===l?"("+f()+" || "+f()+")":2===l?i.pop()+" "+i.pop():o(l))}return f();function f(){return i.pop().replace(/^\((.+)\)$/,"$1")}},n=(e,r)=>{if(0 in e){r=t(r);var a=e[0],o=a<0;o&&(a=-a-1);for(var i=0,l=1,f=!0;;l++,i++){var u,d,s=l<e.length?(typeof e[l])[0]:"";if(i>=r.length||"o"==(d=(typeof(u=r[i]))[0]))return!f||("u"==s?l>a&&!o:""==s!=o);if("u"==d){if(!f||"u"!=s)return!1}else if(f)if(s==d)if(l<=a){if(u!=e[l])return!1}else{if(o?u>e[l]:u<e[l])return!1;u!=e[l]&&(f=!1)}else if("s"!=s&&"n"!=s){if(o||l<=a)return!1;f=!1,l--}else{if(l<=a||d<s!=o)return!1;f=!1}else"s"!=s&&"n"!=s&&(f=!1,l--)}}var c=[],b=c.pop.bind(c);for(i=1;i<e.length;i++){var p=e[i];c.push(1==p?b()|b():2==p?b()&b():p?n(p,r):!b())}return!!b()},i=(e,r)=>{var t=j.S[e];if(!t||!j.o(t,r))throw new Error("Shared module "+r+" doesn't exist in shared scope "+e);return t},l=(e,r)=>{var t=e[r];return(r=Object.keys(t).reduce(((e,r)=>!e||a(e,r)?r:e),0))&&t[r]},f=(e,r)=>{var t=e[r];return Object.keys(t).reduce(((e,r)=>!e||!t[e].loaded&&a(e,r)?r:e),0)},u=(e,r,t,a)=>"Unsatisfied version "+t+" from "+(t&&e[r][t].from)+" of shared singleton module "+r+" (required "+o(a)+")",d=(e,r,t,a)=>{var o=f(e,t);return n(a,o)||c(u(e,t,o,a)),b(e[t][o])},s=(e,r,t)=>{var o=e[r];return(r=Object.keys(o).reduce(((e,r)=>!n(t,r)||e&&!a(e,r)?e:r),0))&&o[r]},c=e=>{"undefined"!=typeof console&&console.warn&&console.warn(e)},b=e=>(e.loaded=1,e.get()),h=(p=e=>function(r,t,a,o){var n=j.I(r);return n&&n.then?n.then(e.bind(e,r,j.S[r],t,a,o)):e(r,j.S[r],t,a,o)})(((e,r,t,a)=>r&&j.o(r,t)?b(l(r,t)):a())),m=p(((e,r,t,a)=>(i(e,t),d(r,0,t,a)))),v=p(((e,r,t,a,o)=>{var n=r&&j.o(r,t)&&s(r,t,a);return n?b(n):o()})),g={},y={6512:()=>m("default","react",[1,18,2,0]),2324:()=>v("default","@emotion/react",[1,11,4,1],(()=>Promise.all([j.e(944),j.e(28),j.e(132)]).then((()=>()=>j(1028))))),3176:()=>v("default","@emotion/styled",[1,11,3,0],(()=>Promise.all([j.e(156),j.e(912),j.e(816)]).then((()=>()=>j(156))))),1708:()=>m("default","@jupyterlab/settingregistry",[1,4,1,0]),2292:()=>m("default","@lumino/signaling",[1,2,0,0]),2508:()=>m("default","@lumino/disposable",[1,2,0,0]),3440:()=>m("default","@jupyterlab/coreutils",[1,6,1,0]),3908:()=>m("default","@jupyterlab/apputils",[1,4,2,0]),5176:()=>v("default","@mui/material",[1,5,15,7],(()=>Promise.all([j.e(944),j.e(448),j.e(912),j.e(584)]).then((()=>()=>j(9448))))),9008:()=>m("default","@jupyterlab/services",[1,7,1,0]),5912:()=>h("default","@emotion/react",(()=>Promise.all([j.e(944),j.e(28),j.e(132)]).then((()=>()=>j(1028))))),2816:()=>v("default","@emotion/react",[1,11,0,0,,"rc",0],(()=>Promise.all([j.e(944),j.e(28)]).then((()=>()=>j(1028))))),584:()=>m("default","react-dom",[1,18,2,0])},w={104:[1708,2292,2508,3440,3908,5176,9008],512:[6512],584:[584],775:[2324,3176],816:[2816],912:[5912]},P={},j.f.consumes=(e,r)=>{j.o(w,e)&&w[e].forEach((e=>{if(j.o(g,e))return r.push(g[e]);if(!P[e]){var t=r=>{g[e]=0,j.m[e]=t=>{delete j.c[e],t.exports=r()}};P[e]=!0;var a=r=>{delete g[e],j.m[e]=t=>{throw delete j.c[e],r}};try{var o=y[e]();o.then?r.push(g[e]=o.then(t).catch(a)):t(o)}catch(e){a(e)}}}))},(()=>{var e={488:0};j.f.j=(r,t)=>{var a=j.o(e,r)?e[r]:void 0;if(0!==a)if(a)t.push(a[2]);else if(/^([59]12|584|775|816)$/.test(r))e[r]=0;else{var o=new Promise(((t,o)=>a=e[r]=[t,o]));t.push(a[2]=o);var n=j.p+j.u(r),i=new Error;j.l(n,(t=>{if(j.o(e,r)&&(0!==(a=e[r])&&(e[r]=void 0),a)){var o=t&&("load"===t.type?"missing":t.type),n=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+o+": "+n+")",i.name="ChunkLoadError",i.type=o,i.request=n,a[1](i)}}),"chunk-"+r,r)}};var r=(r,t)=>{var a,o,[n,i,l]=t,f=0;if(n.some((r=>0!==e[r]))){for(a in i)j.o(i,a)&&(j.m[a]=i[a]);l&&l(j)}for(r&&r(t);f<n.length;f++)o=n[f],j.o(e,o)&&e[o]&&e[o][0](),e[o]=0},t=self.webpackChunkploomber_extension=self.webpackChunkploomber_extension||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})(),j.nc=void 0;var E=j(2608);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB)["ploomber-extension"]=E})();