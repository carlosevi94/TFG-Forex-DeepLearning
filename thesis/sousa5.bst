%%
%% This is file `sousa4.bst',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% merlin.mbs  (with options: `head,exlang,cite,lang,annote,vonx,nm-revv1,jnrlst,nmdash,nmft,nmft-sc,fnm-rm,nmand-rm,dt-jnl,yr-com,yrpp-xsp,tit-qq,qt-g,atit-u,trnum-it,vnum-nr,volp-com,jdt-pc,pp-last,num-xser,pub-date,pub-xpar,pre-pub,edparxc,edby-par,edbyy,spanish,ppx,ed,abr,xedn,and-com,and-com-ed,etal-it,revdata,eprint,url,url-blk,em-it,nfss,{}')
%% spanish.mbs  (with options: `exlang,cite,lang,annote,vonx,nm-revv1,jnrlst,nmdash,nmft,nmft-sc,fnm-rm,nmand-rm,dt-jnl,yr-com,yrpp-xsp,tit-qq,qt-g,atit-u,trnum-it,vnum-nr,volp-com,jdt-pc,pp-last,num-xser,pub-date,pub-xpar,pre-pub,edparxc,edby-par,edbyy,spanish,ppx,ed,abr,xedn,and-com,and-com-ed,etal-it,revdata,eprint,url,url-blk,em-it,nfss,{}')
%% physjour.mbs  (with options: `exlang,cite,lang,annote,vonx,nm-revv1,jnrlst,nmdash,nmft,nmft-sc,fnm-rm,nmand-rm,dt-jnl,yr-com,yrpp-xsp,tit-qq,qt-g,atit-u,trnum-it,vnum-nr,volp-com,jdt-pc,pp-last,num-xser,pub-date,pub-xpar,pre-pub,edparxc,edby-par,edbyy,spanish,ppx,ed,abr,xedn,and-com,and-com-ed,etal-it,revdata,eprint,url,url-blk,em-it,nfss,{}')
%% geojour.mbs  (with options: `exlang,cite,lang,annote,vonx,nm-revv1,jnrlst,nmdash,nmft,nmft-sc,fnm-rm,nmand-rm,dt-jnl,yr-com,yrpp-xsp,tit-qq,qt-g,atit-u,trnum-it,vnum-nr,volp-com,jdt-pc,pp-last,num-xser,pub-date,pub-xpar,pre-pub,edparxc,edby-par,edbyy,spanish,ppx,ed,abr,xedn,and-com,and-com-ed,etal-it,revdata,eprint,url,url-blk,em-it,nfss,{}')
%% photjour.mbs  (with options: `exlang,cite,lang,annote,vonx,nm-revv1,jnrlst,nmdash,nmft,nmft-sc,fnm-rm,nmand-rm,dt-jnl,yr-com,yrpp-xsp,tit-qq,qt-g,atit-u,trnum-it,vnum-nr,volp-com,jdt-pc,pp-last,num-xser,pub-date,pub-xpar,pre-pub,edparxc,edby-par,edbyy,spanish,ppx,ed,abr,xedn,and-com,and-com-ed,etal-it,revdata,eprint,url,url-blk,em-it,nfss,{}')
%% merlin.mbs  (with options: `tail,exlang,cite,lang,annote,vonx,nm-revv1,jnrlst,nmdash,nmft,nmft-sc,fnm-rm,nmand-rm,dt-jnl,yr-com,yrpp-xsp,tit-qq,qt-g,atit-u,trnum-it,vnum-nr,volp-com,jdt-pc,pp-last,num-xser,pub-date,pub-xpar,pre-pub,edparxc,edby-par,edbyy,spanish,ppx,ed,abr,xedn,and-com,and-com-ed,etal-it,revdata,eprint,url,url-blk,em-it,nfss,{}')
%% ----------------------------------------
%% *** bibliografia de JoseRa para escribir en castellano ***
%% 
%% Copyright 1994-2000 Patrick W Daly
 % ===============================================================
 % IMPORTANT NOTICE:
 % This bibliographic style (bst) file has been generated from one or
 % more master bibliographic style (mbs) files, listed above.
 %
 % This generated file can be redistributed and/or modified under the terms
 % of the LaTeX Project Public License Distributed from CTAN
 % archives in directory macros/latex/base/lppl.txt; either
 % version 1 of the License, or any later version.
 % ===============================================================
 % Name and version information of the main mbs file:
 % \ProvidesFile{merlin.mbs}[2000/05/04 4.01 (PWD, AO, DPC)]
 %   For use with BibTeX version 0.99a or later
 %-------------------------------------------------------------------
 % This bibliography style file is intended for texts in
 %   SPANISH
 % This is a special pseudo-numerical bibliography style, intended to be
 %   used to list all the entries in a bib database
 % It requires no extra package to interface to the main text.
 % The form of the \bibitem entries is
 %   \bibitem[key]{key}...
 %---------------------------------------------------------------------

ENTRY
  { address
    annote
    archive
    author
    booktitle
    chapter
    collaboration
    edition
    editor
    eid
    eprint
    howpublished
    institution
    journal
    key
    language
    month
    note
    number
    numpages
    organization
    pages
    publisher
    school
    series
    title
    type
    url
    volume
    year
  }
  {}
  { label }
INTEGERS { output.state before.all mid.sentence after.sentence after.block }
FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}
STRINGS { s t}
FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}
FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}
FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}
FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
  annote missing$
    { "\bibAnnoteFile{" cite$ * "}" * write$ newline$ }
    { "\bibAnnote{" cite$ * "}{" * annote * "}" * write$ newline$
    }
  if$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}
FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}
FUNCTION {add.blank}
{  " " * before.all 'output.state :=
}

FUNCTION {no.blank.or.punct}
{  "\hspace{0pt}" * before.all 'output.state :=
}

FUNCTION {date.block}
{
  new.block
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}
FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}
FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}
FUNCTION {non.stop}
{ duplicate$
   "}" * add.period$
   #-1 #1 substring$ "." =
}

FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}
FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}
FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}
FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}
FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\textit{" swap$ * "}" * }
  if$
}
FUNCTION {scaps}
{ duplicate$ empty$
    { pop$ "" }
    { "\textsc{" swap$ * "}" * }
  if$
}
FUNCTION {bib.name.font}
{ scaps }
FUNCTION {bib.fname.font}
{ skip$ }
FUNCTION {tie.or.space.prefix}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$
}

FUNCTION {capitalize}
{ "u" change.case$ "t" change.case$ }

FUNCTION {space.word}
{ " " swap$ * " " * }
 % Here are the language-specific definitions for explicit words.
 % Each function has a name bbl.xxx where xxx is the English word.
 %-------------------------------------------------------------------
 % Begin module:
 % \ProvidesFile{spanish.mbs}[2000/03/15 4.0 (PWD/TB)]

 % The language selected here is SPANISH
FUNCTION {bbl.and}
{ "y"}

FUNCTION {bbl.etal}
{ "et~al." }

FUNCTION {bbl.editors}
{ "eds." }

FUNCTION {bbl.editor}
{ "ed." }

FUNCTION {bbl.edby}
{ "editado por" }

FUNCTION {bbl.edition}
{ "ed\raise1ex\hbox{\underbar{\scriptsize \'on}}." }

FUNCTION {bbl.volume}
{ "tomo" }

FUNCTION {bbl.of}
{ "de" }

FUNCTION {bbl.number}
{ "n\raise1ex\hbox{\underbar{\scriptsize o}}" }

FUNCTION {bbl.nr}
{ "n\raise1ex\hbox{\underbar{\scriptsize o}}" }

FUNCTION {bbl.in}
{ "en" }

FUNCTION {bbl.pages}
{ "p\'aginas" }

FUNCTION {bbl.page}
{ "p\'agina" }

FUNCTION {bbl.eidpp}
{ "p\'aginas" }

FUNCTION {bbl.chapter}
{ "cap." }

FUNCTION {bbl.techrep}
{ "Inf.\ T\'ec." }

FUNCTION {bbl.mthesis}
{ "Proyecto Fin de Carrera" }

FUNCTION {bbl.phdthesis}
{ "Tesis Doctoral" }

MACRO {jan} {"ene."}

MACRO {feb} {"feb."}

MACRO {mar} {"mar."}

MACRO {apr} {"abr."}

MACRO {may} {"mayo"}

MACRO {jun} {"jun."}

MACRO {jul} {"jul."}

MACRO {aug} {"ago."}

MACRO {sep} {"sep."}

MACRO {oct} {"oct."}

MACRO {nov} {"nov."}

MACRO {dec} {"dic."}

 % End module: spanish.mbs

























 %-------------------------------------------------------------------

 % Begin module:


 % \ProvidesFile{physjour.mbs}[1999/10/05 2.1 (PWD)]








































































































































MACRO {aa}{"Astronomy and Astrophysics"}

MACRO {aasup}{"Astronomy and Astrophysics, Supplement Series"}

MACRO {aj}{"Astronomical Journal"}

MACRO {aph} {"Acta Physica"}

MACRO {advp} {"Advances in Physics"}

MACRO {ajp} {"American Journal of Physics"}

MACRO {ajm} {"American Journal of Mathematics"}

MACRO {amsci} {"American Scientist"}

MACRO {anofd} {"Annals of Fluid Dynamics"}

MACRO {am} {"Annals of Mathematics"}

MACRO {ap} {"Annals of Physics"}

MACRO {adp} {"Annalen der Physik"}

MACRO {ao} {"Applied Optics"}

MACRO {apl} {"Applied Physics Letters"}

MACRO {app} {"Astroparticle Physics"}

MACRO {apj} {"Astrophysical Journal"}

MACRO {apjsup} {"Astrophysical Journal Supplement"}

MACRO {apss} {"Astrophysics and Space Science"}

MACRO {araa} {"Annual Review of Astronomy and Astrophysics"}

MACRO {baas} {"Bulletin of the American Astronomical Society"}

MACRO {baps} {"Bulletin of the American Physical Society"}

MACRO {cmp} {"Communications in Mathematical Physics"}

MACRO {cpam} {"Communications on Pure and Applied Mathematics"}

MACRO {cppcf} {"Comments on Plasma Physics and Controlled Fusion"}

MACRO {cpc} {"Computer Physics Communications"}

MACRO {cqg} {"Classical and Quantum Gravity"}

MACRO {cra} {"Comptes Rendus des S{\'e}ances de l'Acad{\'e}mie des Sciences A"}

MACRO {fed} {"Fusion Engineering and Design"}

MACRO {ft} {"Fusion Technology"}

MACRO {grg} {"General Relativity and Gravitation"}

MACRO {ieeens} {"IEEE Transactions on Nuclear Science"}

MACRO {ieeeps} {"IEEE Transactions on Plasma Science"}

MACRO {ijimw} {"International Journal of Infrared and Millimeter Waves"}

MACRO {ip} {"Infrared Physics"}

MACRO {irp} {"Infrared Physics"}

MACRO {jap} {"Journal of Applied Physics"}

MACRO {jasa} {"Journal of the Acoustical Society of America"}

MACRO {jcp} {"Journal of Computational Physics"}

MACRO {jetp} {"Soviet Physics--JETP"}

MACRO {jfe} {"Journal of Fusion Energy"}

MACRO {jfm} {"Journal of Fluid Mechanics"}

MACRO {jmp} {"Journal of Mathematical Physics"}

MACRO {jne} {"Journal of Nuclear Energy"}

MACRO {jnec} {"Journal of Nuclear Energy, Part C: Plasma Physics, Accelerators, Thermonuclear Research"}

MACRO {jnm} {"Journal of Nuclear Materials"}

MACRO {jpc} {"Journal of Physical Chemistry"}

MACRO {jpp} {"Journal of Plasma Physics"}

MACRO {jpsj} {"Journal of the Physical Society of Japan"}

MACRO {jsi} {"Journal of Scientific Instruments"}

MACRO {jvst} {"Journal of Vacuum Science and Technology"}

MACRO {nat} {"Nature"}

MACRO {nature} {"Nature"}

MACRO {nedf} {"Nuclear Engineering and Design/Fusion"}

MACRO {nf} {"Nuclear Fusion"}

MACRO {nim} {"Nuclear Instruments and Methods"}

MACRO {nimpr} {"Nuclear Instruments and Methods in Physics Research"}

MACRO {np} {"Nuclear Physics"}

MACRO {npb} {"Nuclear Physics B"}

MACRO {nt/f} {"Nuclear Technology/Fusion"}

MACRO {npbpc} {"Nuclear Physics B (Proceedings Suppluments)"}

MACRO {inc} {"Il Nuovo Cimento"}

MACRO {nc} {"Nuovo Cimento"}

MACRO {pf} {"Physics of Fluids"}

MACRO {pfa} {"Physics of Fluids A: Fluid Dynamics"}

MACRO {pfb} {"Physics of Fluids B: Plasma Physics"}

MACRO {pl} {"Physics Letters"}

MACRO {pla} {"Physics Letters A"}

MACRO {plb} {"Physics Letters B"}

MACRO {prep} {"Physics Reports"}

MACRO {pnas} {"Proceedings of the National Academy of Sciences of the USA"}

MACRO {pp} {"Physics of Plasmas"}

MACRO {ppcf} {"Plasma Physics and Controlled Fusion"}

MACRO {phitrsl} {"Philosophical Transactions of the Royal Society of London"}

MACRO {prl} {"Physical Review Letters"}

MACRO {pr} {"Physical Review"}

MACRO {physrev} {"Physical Review"}

MACRO {pra} {"Physical Review A"}

MACRO {prb} {"Physical Review B"}

MACRO {prc} {"Physical Review C"}

MACRO {prd} {"Physical Review D"}

MACRO {pre} {"Physical Review E"}

MACRO {ps} {"Physica Scripta"}

MACRO {procrsl} {"Proceedings of the Royal Society of London"}

MACRO {rmp} {"Reviews of Modern Physics"}

MACRO {rsi} {"Review of Scientific Instruments"}

MACRO {science} {"Science"}

MACRO {sciam} {"Scientific American"}

MACRO {sam} {"Studies in Applied Mathematics"}

MACRO {sjpp} {"Soviet Journal of Plasma Physics"}

MACRO {spd} {"Soviet Physics--Doklady"}

MACRO {sptp} {"Soviet Physics--Technical Physics"}

MACRO {spu} {"Soviet Physics--Uspeki"}

MACRO {st} {"Sky and Telescope"}



 % End module: physjour.mbs



 %-------------------------------------------------------------------
 % Begin module:
 % \ProvidesFile{geojour.mbs}[1999/11/16 2.0g (PWD)]
MACRO {aisr} {"Advances in Space Research"}
MACRO {ag} {"Annales Geophysic\ae"}
MACRO {anigeo} {"Annali di Geofisica"}
MACRO {angl} {"Annals of Glaciology"}
MACRO {andmet} {"Annalen der Meteorologie"}
MACRO {andgeo} {"Annales de Geophysique"}
MACRO {andphy} {"Annales de Physique"}
MACRO {afmgb} {"Archiv f\^^b{u}r Meteorologie, Geophysik und Bioklimatologie"}
MACRO {ass}{"Astrophysics and Space Science"}
MACRO {atph} {"Atm\'osphera"}
MACRO {aao} {"Atmosphere-Ocean"}
MACRO {atenv} {"Atmospheric Environment"}
MACRO {aujag} {"Australian Journal of Agricultural Research"}
MACRO {aumet} {"Australian Meteorological Magazine"}
MACRO {blmet} {"Boundary-Layer Meteorology"}
MACRO {bams} {"Bulletin of the American Meteorological Society"}
MACRO {cch} {"Climatic Change"}
MACRO {cdyn} {"Climate Dynamics"}
MACRO {cbul} {"Climatological Bulletin"}
MACRO {cap} {"Contributions to Atmospheric Physics"}
MACRO {dsr} {"Deep-Sea Research"}
MACRO {dhz} {"Deutsche Hydrographische Zeitschrift"}
MACRO {dao} {"Dynamics of Atmospheres and Oceans"}
MACRO {eco} {"Ecology"}
MACRO {empl}{"Earth, Moon and Planets"}
MACRO {envres} {"Environmental Research"}
MACRO {envst} {"Environmental Science and Technology"}
MACRO {ecms} {"Estuarine and Coastal Marine Science"}
MACRO {expa}{"Experimental Astronomy"}
MACRO {geoint} {"Geofisica Internationale"}
MACRO {geopub} {"Geofysiske Publikasjoner"}
MACRO {geogeo} {"Geologiya i Geofizika"}
MACRO {gafd} {"Geophysical and Astrophysical Fluid Dynamics"}
MACRO {gfd} {"Geophysical Fluid Dynamics"}
MACRO {geomag} {"Geophysical Magazine"}
MACRO {georl} {"Geophysical Research Letters"}
MACRO {grl} {"Geophysical Research Letters"}
MACRO {ga} {"Geophysica"}
MACRO {gs} {"Geophysics"}
MACRO {ieeetap} {"IEEE Transactions on Antennas and Propagation"}
MACRO {ijawp} {"International Journal of Air and Water Pollution"}
MACRO {ijc} {"International Journal of Climatology"}
MACRO {ijrs} {"International Journal of Remote Sensing"}
MACRO {jam} {"Journal of Applied Meteorology"}
MACRO {jaot} {"Journal of Atmospheric and Oceanic Technology"}
MACRO {jatp} {"Journal of Atmospheric and Terrestrial Physics"}
MACRO {jce} {"Journal of Climate"}
MACRO {jcam} {"Journal of Climate and Applied Meteorology"}
MACRO {jcm} {"Journal of Climate Meteorology"}
MACRO {jcy} {"journal of Climatology"}
MACRO {jgr} {"Journal of Geophysical Research"}
MACRO {jga} {"Journal of Glaciology"}
MACRO {jh} {"Journal of Hydrology"}
MACRO {jmr} {"Journal of Marine Research"}
MACRO {jmrj} {"Journal of Meteorological Research, Japan"}
MACRO {jm} {"Journal of Meteorology"}
MACRO {jpo} {"Journal of Physical Oceanography"}
MACRO {jra} {"Journal de Recherches Atmospheriques"}
MACRO {jaes} {"Journal of the Aeronautical Sciences"}
MACRO {japca} {"Journal of the Air Pollution Control Association"}
MACRO {jas} {"Journal of the Atmospheric Sciences"}
MACRO {jmts} {"Journal of the Marine Technology Society"}
MACRO {jmsj} {"Journal of Meteorological Society of Japan"}
MACRO {josj} {"Journal of the Oceanographical Society of Japan"}
MACRO {jwm} {"Journal of Weather Modification"}
MACRO {lao} {"Limnology and Oceanography"}
MACRO {mwl} {"Mariners Weather Log"}
MACRO {mau} {"Mausam"}
MACRO {meteor} {"``Meteor'' Forschungsergebnisse"}
MACRO {map} {"Meteorology and Atmospheric Physics"}
MACRO {metmag} {"Meteorological Magazine"}
MACRO {metmon} {"Meteorological Monographs"}
MACRO {metrun} {"Meteorologische Rundschau"}
MACRO {metzeit} {"Meteorologische Zeitschrift"}
MACRO {metgid} {"Meteorologiya i Gidrologiya"}
MACRO {mwr} {"Monthly Weather Review"}
MACRO {nwd} {"National Weather Digest"}
MACRO {nzjmfr} {"New Zealand Journal of Marine and Freshwater Research"}
MACRO {npg} {"Nonlinear Processes in Geophysics"}
MACRO {om} {"Oceanography and Meteorology"}
MACRO {ocac} {"Oceanologica Acta"}
MACRO {oceanus} {"Oceanus"}
MACRO {paleoc} {"Paleoceanography"}
MACRO {pce} {"Physics and Chemistry of the Earth"}
MACRO {pmg} {"Papers in Meteorology and Geophysics"}
MACRO {ppom} {"Papers in Physical Oceanography and Meteorology"}
MACRO {physzeit} {"Physikalische Zeitschrift"}
MACRO {pps} {"Planetary and Space Science"}
MACRO {pss} {"Planetary and Space Science"}
MACRO {pag} {"Pure and Applied Geophysics"}
MACRO {qjrms} {"Quarterly Journal of the Royal Meteorological Society"}
MACRO {quatres} {"Quaternary Research"}
MACRO {rsci} {"Radio Science"}
MACRO {rse} {"Remote Sensing of the Environment"}
MACRO {rgeo} {"Reviews of Geophysics"}
MACRO {rgsp} {"Reviews of Geophysics and Space Physics"}
MACRO {rdgeo} {"Revista de Geofisica"}
MACRO {revmeta} {"Revista Meteorologica"}
MACRO {sgp}{"Surveys in Geophysics"}
MACRO {sp} {"Solar Phys."}
MACRO {ssr} {"Space Science Review"}
MACRO {tellus} {"Tellus"}
MACRO {tac} {"Theoretical and Applied Climatology"}
MACRO {tagu} {"Transactions, American Geophysical Union (EOS)"}
MACRO {wrr} {"Water Resources Research"}
MACRO {weather} {"Weather"}
MACRO {wafc} {"Weather and Forecasting"}
MACRO {ww} {"Weatherwise"}
MACRO {wmob} {"World Meteorological Organization Bulletin"}
MACRO {zeitmet} {"Zeitschrift f\^^b{u}r Meteorologie"}
 % End module: geojour.mbs

























 %-------------------------------------------------------------------

 % Begin module:


 % \ProvidesFile{photjour.mbs}[1999/02/24 2.0b (PWD)]









































































































MACRO {appopt} {"Applied Optics"}

MACRO {bell} {"Bell Systems Technical Journal"}

MACRO {ell} {"Electronics Letters"}

MACRO {jasp} {"Journal of Applied Spectroscopy"}

MACRO {jqe} {"IEEE Journal of Quantum Electronics"}

MACRO {jlwt} {"Journal of Lightwave Technology"}

MACRO {jmo} {"Journal of Modern Optics"}

MACRO {josa} {"Journal of the Optical Society of America"}

MACRO {josaa} {"Journal of the Optical Society of America~A"}

MACRO {josab} {"Journal of the Optical Society of America~B"}

MACRO {jdp} {"Journal de Physique"}

MACRO {oc} {"Optics Communications"}

MACRO {ol} {"Optics Letters"}

MACRO {phtl} {"IEEE Photonics Technology Letters"}

MACRO {pspie} {"Proceedings of the Society for Photo-Optical Instruments Engineers"}

MACRO {sse} {"Solid-State Electronics"}

MACRO {sjot} {"Soviet Journal of Optical Technology"}

MACRO {sjqe} {"Soviet Journal of Quantum Electronics"}

MACRO {sleb} {"Soviet Physics--Lebedev Institute Reports"}

MACRO {stph} {"Soviet Physics--Technical Physics"}

MACRO {stphl} {"Soviet Technical Physics Letters"}

MACRO {vr} {"Vision Research"}

MACRO {zph} {"Zeitschrift f\^^b{u}r Physik"}

MACRO {zphb} {"Zeitschrift f\^^b{u}r Physik~B"}

MACRO {zphd} {"Zeitschrift f\^^b{u}r Physik~D"}

MACRO {CLEO} {"Conf. on Lasers and Electro-Optics"}

MACRO {ASSL} {"Advanced Solid State Lasers"}

MACRO {OSA}  {"Optical Society of America"}



 % End module: photjour.mbs



%% Copyright 1994-2000 Patrick W Daly
MACRO {acmcs} {"ACM Computing Surveys"}

MACRO {acta} {"Acta Informatica"}

MACRO {cacm} {"Communications of the ACM"}

MACRO {ibmjrd} {"IBM Journal of Research and Development"}

MACRO {ibmsj} {"IBM Systems Journal"}

MACRO {ieeese} {"IEEE Transactions on Software Engineering"}

MACRO {ieeetc} {"IEEE Transactions on Computers"}

MACRO {ieeetcad}
 {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}

MACRO {ipl} {"Information Processing Letters"}

MACRO {jacm} {"Journal of the ACM"}

MACRO {jcss} {"Journal of Computer and System Sciences"}

MACRO {scp} {"Science of Computer Programming"}

MACRO {sicomp} {"SIAM Journal on Computing"}

MACRO {tocs} {"ACM Transactions on Computer Systems"}

MACRO {tods} {"ACM Transactions on Database Systems"}

MACRO {tog} {"ACM Transactions on Graphics"}

MACRO {toms} {"ACM Transactions on Mathematical Software"}

MACRO {toois} {"ACM Transactions on Office Information Systems"}

MACRO {toplas} {"ACM Transactions on Programming Languages and Systems"}

MACRO {tcs} {"Theoretical Computer Science"}
FUNCTION {bibinfo.check}
{ swap$
  duplicate$ missing$
    {
      pop$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ pop$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}
FUNCTION {bibinfo.warn}
{ swap$
  duplicate$ missing$
    {
      swap$ "missing " swap$ * " in " * cite$ * warning$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ "empty " swap$ * " in " * cite$ * warning$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}
FUNCTION {format.eprint}
{ eprint duplicate$ empty$
    'skip$
    { "\eprint"
      archive empty$
        'skip$
        { "[" * archive * "]" * }
      if$
      "{" * swap$ * "}" *
    }
  if$
}
FUNCTION {format.url}
{ url empty$
    { "" }
    { "\urlprefix\url{" url * "}" * }
  if$
}

STRINGS  { bibinfo}
INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 'bibinfo :=
  duplicate$ empty$ 'skip$ {
  's :=
  "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      nameptr #1 >
        { "{ff}" format.name$ duplicate$ empty$ 'skip$
            { tie.or.space.prefix bib.fname.font swap$ * }
          if$
          s nameptr
          "{vv~}{ll}" format.name$ bib.name.font *
          s nameptr
          "{jj}" format.name$ duplicate$ empty$ 'skip$
            { bib.fname.font ", " swap$ * }
          if$
        }
        { "{vv~}{ll}" format.name$ bib.name.font
          s nameptr
          "{ff}{, jj}"
          format.name$ duplicate$ empty$ 'skip$
            { bib.fname.font ", " swap$ * }
          if$
        }
      if$
      *
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            {
              "," *
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                {
                  " " * bbl.etal emphasize *
                }
                {
                  bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  } if$
}
FUNCTION {format.names.ed}
{
  'bibinfo :=
  duplicate$ empty$ 'skip$ {
  's :=
  "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{ff~}{vv~}{ll}{, jj}"
      format.name$
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            {
              "," *
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                {

                  " " * bbl.etal emphasize *
                }
                {
                  bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  } if$
}
FUNCTION {format.authors}
{ author "author" format.names
    duplicate$ empty$ 'skip$
    { collaboration "collaboration" bibinfo.check
      duplicate$ empty$ 'skip$
        { " (" swap$ * ")" * }
      if$
      *
    }
  if$
}
FUNCTION {get.bbl.editor}
{ editor num.names$ #1 > 'bbl.editors 'bbl.editor if$ }

FUNCTION {format.editors}
{ editor "editor" format.names duplicate$ empty$ 'skip$
    {
      " " *
      get.bbl.editor
   "(" swap$ * ")" *
      *
    }
  if$
}
FUNCTION {select.language}
{ duplicate$ empty$
    'pop$
    { language empty$
        'skip$
        { "{\selectlanguage{" language * "}" * swap$ * "}" * }
      if$
    }
    if$
}

FUNCTION {format.note}
{
 note empty$
    { "" }
    { note #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          { "l" }
          { "u" }
        if$
        change.case$
        }
      if$
      note #2 global.max$ substring$ * "note" bibinfo.check
    }
  if$
}

FUNCTION {format.title}
{ title
  "title" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      "\enquote{" swap$ *
      add.period$ "}" *
      select.language
    }
  if$
}
FUNCTION {output.bibitem}
{ newline$
  "\bibitem[" label * "]{" * write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {n.dashify}
{
  't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {word.in}
{ bbl.in capitalize
  " " * }

FUNCTION {format.date}
{
  month "month" bibinfo.check
  duplicate$ empty$
  year  "year"  bibinfo.check duplicate$ empty$
    { swap$ 'skip$
        { "there's a month but no year in " cite$ * warning$ }
      if$
      *
    }
    { swap$ 'skip$
        {
          swap$
          " " * swap$
        }
      if$
      *
    }
  if$
  duplicate$ empty$
    'skip$
    {
      before.all 'output.state :=
    ", " swap$ *
    }
  if$
}
FUNCTION{format.year}
{ year "year" bibinfo.check duplicate$ empty$
    {  "empty year in " cite$ *
       warning$
    }
    {
  "(" swap$ * ")" *
    }
  if$
}
FUNCTION {format.btitle}
{ title "title" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      emphasize
      select.language
    }
  if$
}
FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}
FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { bbl.volume volume tie.or.space.prefix
      "volume" bibinfo.check * *
      series "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { swap$ bbl.of space.word * swap$
          emphasize * }
      if$
      "volume and number" number either.or.check
    }
  if$
}
FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series field.or.null }
        { series empty$
            { number "number" bibinfo.check }
        { output.state mid.sentence =
            { bbl.number }
            { bbl.number capitalize }
          if$
          number tie.or.space.prefix "number" bibinfo.check * *
          bbl.in space.word *
          series "series" bibinfo.check *
        }
      if$
    }
      if$
    }
    { "" }
  if$
}

FUNCTION {format.edition}
{ edition duplicate$ empty$ 'skip$
    {
      output.state mid.sentence =
        { "l" }
        { "t" }
      if$ change.case$
      "edition" bibinfo.check
      " " * bbl.edition *
    }
  if$
}
INTEGERS { multiresult }
FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}
FUNCTION {format.pages}
{ pages duplicate$ empty$ 'skip$
    { duplicate$ multi.page.check
        {
          n.dashify
        }
        {
        }
      if$
      "pages" bibinfo.check
    }
  if$
}
FUNCTION {format.journal.pages}
{ pages duplicate$ empty$ 'pop$
    { swap$ duplicate$ empty$
        { pop$ pop$ format.pages }
        {
          ", " *
          format.year * ", " *
          swap$
          n.dashify
          "pages" bibinfo.check
          *
        }
      if$
    }
  if$
}
FUNCTION {format.journal.eid}
{ eid "eid" bibinfo.check
  duplicate$ empty$ 'skip$
    { swap$ duplicate$ empty$ 'skip$
      {
          ", " *
      }
      if$
      swap$
      numpages empty$ 'skip$
        { bbl.eidpp numpages tie.or.space.prefix
          "numpages" bibinfo.check * *
          " (" swap$ * ")" * *
        }
      if$
    }
  if$ *
}
FUNCTION {format.vol.num.pages}
{ volume field.or.null
  duplicate$ empty$ 'skip$
    {
      "volume" bibinfo.check
    }
  if$
  number "number" bibinfo.check duplicate$ empty$ 'skip$
    {
      swap$ duplicate$ empty$
        { "there's a number but no volume in " cite$ * warning$ }
        'skip$
      if$
      swap$
      ", " bbl.nr * number tie.or.space.prefix pop$ * swap$ *
    }
  if$ *
}

FUNCTION {format.chapter.pages}
{ chapter empty$
    { "" }
    { type empty$
        { bbl.chapter }
        { type "l" change.case$
          "type" bibinfo.check
        }
      if$
      chapter tie.or.space.prefix
      "chapter" bibinfo.check
      * *
    }
  if$
}

FUNCTION {format.booktitle}
{
  booktitle "booktitle" bibinfo.check
  emphasize
}
FUNCTION {format.in.ed.booktitle}
{ format.booktitle duplicate$ empty$ 'skip$
    {
      editor "editor" format.names.ed duplicate$ empty$ 'pop$
        {
          get.bbl.editor
          swap$ "," *
          " " * swap$ *
          "(" swap$ * ")" *
          swap$
          " " * swap$
          * }
      if$
      word.in swap$ *
    }
  if$
}
FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
  key empty$ not and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}
FUNCTION {format.thesis.type}
{ type duplicate$ empty$
    'pop$
    { swap$ pop$
      "t" change.case$ "type" bibinfo.check
    }
  if$
}
FUNCTION {format.tr.number}
{ number "number" bibinfo.check
  type duplicate$ empty$
    { pop$ bbl.techrep }
    'skip$
  if$
  "type" bibinfo.check
  swap$ duplicate$ empty$
    { pop$ "t" change.case$ }
    { tie.or.space.prefix * * }
  if$
}
FUNCTION {format.article.crossref}
{
  key duplicate$ empty$
    { pop$
      journal duplicate$ empty$
        { "need key or journal for " cite$ * " to crossref " * crossref * warning$ }
        { "journal" bibinfo.check emphasize word.in swap$ * }
      if$
    }
    { word.in swap$ * " " *}
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  "editor" bibinfo.check
  editor num.names$ duplicate$
  #2 >
    { pop$
      "editor" bibinfo.check
      " " * bbl.etal
      emphasize
      *
    }
    { #2 <
        'skip$
        { editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            {
              "editor" bibinfo.check
              " " * bbl.etal
              emphasize
              *
            }
            {
             bbl.and space.word
              * editor #2 "{vv~}{ll}" format.name$
              "editor" bibinfo.check
              *
            }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.book.crossref}
{ volume duplicate$ empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      pop$ word.in
    }
    { bbl.volume
      capitalize
      swap$ tie.or.space.prefix "volume" bibinfo.check * * bbl.of space.word *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { series empty$
            { "need editor, key, or series for " cite$ * " to crossref " *
              crossref * warning$
              "" *
            }
            { series emphasize * }
          if$
        }
        { key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.incoll.inproc.crossref}
{
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { format.booktitle duplicate$ empty$
            { "need editor, key, or booktitle for " cite$ * " to crossref " *
              crossref * warning$
            }
            { word.in swap$ * }
          if$
        }
        { word.in key * " " *}
      if$
    }
    { word.in format.crossref.editor * " " *}
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.org.or.pub}
{ 't :=
  ""
  year empty$
    { "empty year in " cite$ * warning$ }
    'skip$
  if$
  address empty$ t empty$ and
  year empty$ and
    'skip$
    {
      t empty$
        { address "address" bibinfo.check *
        }
        { t *
          address empty$
            'skip$
            { ", " * address "address" bibinfo.check * }
          if$
        }
      if$
      year empty$
        'skip$
        { t empty$ address empty$ and
            'skip$
            { ", " * }
          if$
          year "year" bibinfo.check *
        }
      if$
    }
  if$
}
FUNCTION {format.publisher.address}
{ publisher "publisher" bibinfo.warn format.org.or.pub
}

FUNCTION {format.organization.address}
{ organization "organization" bibinfo.check format.org.or.pub
}

STRINGS {oldname}

FUNCTION {name.or.dash}
{ 's :=
   oldname empty$
     { s 'oldname := s }
     { s oldname =
         { "---" }
         { s 'oldname := s }
       if$
     }
   if$
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  name.or.dash
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    {
      journal
      "journal" bibinfo.check
      emphasize
      "journal" output.check
      format.vol.num.pages output
    }
    { format.article.crossref output.nonnull
    }
  if$
  eid empty$
    { format.journal.pages }
    { format.journal.eid }
  if$
  new.block
  format.publisher.address output
  format.note output
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      add.blank
      name.or.dash
    }
    { format.authors output.nonnull
      name.or.dash
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  new.block
  format.btitle "title" output.check
  crossref missing$
    { format.bvolume output
      new.block
      format.number.series output
      new.sentence
      format.publisher.address output
    }
    {
      new.block
      format.book.crossref output.nonnull
      format.date "year" output.check
    }
  if$
  format.edition output
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {booklet}
{ output.bibitem
  format.authors output
  name.or.dash
  new.block
  format.title "title" output.check
  new.block
  howpublished "howpublished" bibinfo.check output
  address "address" bibinfo.check output
  format.date output
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      name.or.dash
    }
    { format.authors output.nonnull
      name.or.dash
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  new.block
  format.btitle "title" output.check
  crossref missing$
    {
      format.publisher.address output
      format.bvolume output
      new.block
      format.number.series output
      new.sentence
    }
    {
      new.block
      format.book.crossref output.nonnull
    }
  if$
  format.edition output
  add.blank
  format.pages "pages" output.check
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  name.or.dash
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.publisher.address output
      format.bvolume output
      format.number.series output
      new.sentence
      format.edition output
    }
    { format.incoll.inproc.crossref output.nonnull
    }
  if$
  format.pages "pages" output.check
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  name.or.dash
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      new.sentence
      publisher empty$
        { format.organization.address output }
        { organization "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
      format.bvolume output
      format.number.series output
    }
    { format.incoll.inproc.crossref output.nonnull
    }
  if$
  format.pages "pages" output.check
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {conference} { inproceedings }
FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization "organization" bibinfo.check
      duplicate$ empty$ 'pop$
        { output
          address "address" bibinfo.check output
        }
      if$
    }
    { format.authors output.nonnull }
  if$
  name.or.dash
  new.block
  format.btitle "title" output.check
  author empty$
    { organization empty$
        {
          address new.block.checka
          address "address" bibinfo.check output
        }
        'skip$
      if$
    }
    {
      organization address new.block.checkb
      organization "organization" bibinfo.check output
      address "address" bibinfo.check output
    }
  if$
  format.edition output
  format.date output
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  name.or.dash
  new.block
  format.btitle
  "title" output.check
  new.block
  bbl.mthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  name.or.dash
  title howpublished new.block.checkb
  format.title output
  howpublished new.block.checka
  howpublished "howpublished" bibinfo.check output
  format.date output
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
  empty.misc.check
}
FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  name.or.dash
  new.block
  format.btitle
  "title" output.check
  new.block
  bbl.phdthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization "organization" bibinfo.check output
    }
    { format.editors output.nonnull }
  if$
  name.or.dash
  new.block
  format.btitle "title" output.check
  format.bvolume output
  format.number.series output
  editor empty$
    { publisher empty$
        'skip$
        {
          new.sentence
          format.publisher.address output
        }
      if$
    }
    { publisher empty$
        {
          new.sentence
          format.organization.address output }
        {
          new.sentence
          organization "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
     }
  if$
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  name.or.dash
  new.block
  format.title
  "title" output.check
  new.block
  format.tr.number emphasize output.nonnull
  institution "institution" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  new.block
  format.note output
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  name.or.dash
  new.block
  format.title "title" output.check
  format.date output
  new.block
  format.note "note" output.check
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {default.type} { misc }
READ
FUNCTION {sortify}
{ purify$
  "l" change.case$
}
INTEGERS { len }
FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}
FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{ll{ }}{  ff{ }}{  jj{ }}"
      format.name$ 't :=
      nameptr #1 >
        {
          "   "  *
          namesleft #1 = t "others" = and
            { "zzzzz" * }
            { t sortify * }
          if$
        }
        { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}
FUNCTION {author.sort}
{ author empty$
    { key empty$
        { "to sort, need author or key in " cite$ * warning$
          ""
        }
        { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { key empty$
            { "to sort, need author, editor, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
        { key empty$
            { "to sort, need author, organization, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
        { key empty$
            { "to sort, need editor, organization, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { editor sort.format.names }
  if$
}
FUNCTION {presort}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.sort
    { type$ "proceedings" =
        'editor.organization.sort
        { type$ "manual" =
            'author.organization.sort
            'author.sort
          if$
        }
      if$
    }
  if$
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}
ITERATE {presort}
SORT
STRINGS { longest.label }
INTEGERS { number.label longest.label.width }
FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}
FUNCTION {longest.label.pass}
{ cite$ 'label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}
EXECUTE {initialize.longest.label}
ITERATE {longest.label.pass}
FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{"  longest.label  * "}" *
  write$ newline$
  "\ProvideTextCommand{\guillemotleft}{OT1}{%"
  write$ newline$
  "  \leavevmode\raise .27ex\hbox{$\scriptscriptstyle\ll$}}"
  write$ newline$
  "\ProvideTextCommand{\guillemotright}{OT1}{%"
  write$ newline$
  "  \leavevmode\raise .27ex\hbox{$\scriptscriptstyle\gg$}}"
  write$ newline$
  "\newcommand{\enquote}[1]{\guillemotleft#1\guillemotright}"
  write$ newline$
  "\expandafter\ifx\csname url\endcsname\relax"
  write$ newline$
  "  \def\url#1{\texttt{#1}}\fi"
  write$ newline$
  "\expandafter\ifx\csname urlprefix\endcsname\relax\def\urlprefix{URL }\fi"
  write$ newline$
  "\providecommand{\selectlanguage}[1]{\relax}"
  write$ newline$
  "\providecommand{\bibAnnoteFile}[1]{%"
  write$ newline$
  "  \IfFileExists{#1}{\begin{quotation}\noindent\textsc{Key:} #1\\"
  write$ newline$
  "  \textsc{Annotation:}\ \input{#1}\end{quotation}}{}}"
  write$ newline$
  "\providecommand{\bibAnnote}[2]{%"
  write$ newline$
  "  \begin{quotation}\noindent\textsc{Key:} #1\\"
  write$ newline$
  "  \textsc{Annotation:}\ #2\end{quotation}}"
  write$ newline$
  "\providecommand{\eprint}[2][]{\url{#2}}"
  write$ newline$
}
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}
FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}
EXECUTE {end.bib}
%% End of customized bst file
%%
%% End of file `sousa4.bst'.
