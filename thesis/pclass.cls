% Version 1.2 Cambios menores por JoséRa Portillo
% Version 1.3 2014-09-23 JoséRa Portillo Adaptación de Trabajo fin de Grado

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pclass}[2008/09/08 PFC class Universidad de Sevilla]
\DeclareOption{openright}{ \PassOptionsToClass{openright}{book} }
\DeclareOption{openany}{ \PassOptionsToClass{openany}{book} }
\DeclareOption{twoside}{ \PassOptionsToClass{twoside}{book} }
\DeclareOption{oneside}{ \PassOptionsToClass{oneside}{book} }
%\ExecuteOptions{twoside,openright}
\ExecuteOptions{oneside}   % NO SE IMPRIME
\ProcessOptions 

\LoadClass[a4paper,11pt]{book}

\textwidth=350pt
\textheight=600pt
\marginparwidth=100pt

%\setlength{\oddsidemargin}{53pt}
%\setlength{\evensidemargin}{53pt}
%\setlength{\marginparwidth}{57pt}
%\setlength{\footskip}{30pt}

\RequirePackage [a4paper,left=3cm,right=3cm,top=2.5cm,bottom=2.5cm]{geometry}
\RequirePackage {graphicx}%para incluir im{\'a}genes
\RequirePackage{amsmath,amsthm,amssymb,latexsym}%s{\'\i}mbolos matem{\'a}ticos
\usepackage{url}%Para direcciones web
\RequirePackage[spanish,activeacute]{babel}%Traducci{\'o}n al espa{\~n}ol

%DIFERENTES TIPOS DE LETRA
%\usepackage{palatino}
%\usepackage{newcent}   
%\usepackage{courier}
\usepackage{helvet}

\usepackage{symbols}%s{\'\i}mbolo matm{\'a}ticos

\usepackage[clearempty,pagestyles,newparttoc]{titlesec}%configuraci{\'o}n de t{\'\i}tulos

\usepackage{titletoc}%configuraci{\'o}n de t{\'\i}tulos

\usepackage{tocbibind}%a{\~n}ade apartados al {\'\i}ndice general
\setcounter{tocdepth}{2}%controlar hasta qu{\'e} nivel se incluye un elemento en la tabla de materias

\usepackage{boxedminipage}%para hacer cuadros vac{\'\i}os
\usepackage{multicol}%insertar tablas
\usepackage{color}%definir colores

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%para c{\'o}digo fuente%%%%%%%%%%%%%%%%%%%%%
\usepackage{listings}
% Definiendo colores para los listados de c{\'o}digo fuente
\definecolor{violet}{rgb}{0.5,0,0.5}
\definecolor{shadow}{rgb}{0.5,0.4,0.5}

\definecolor{hellgelb}{rgb}{1,1,0.8}
\definecolor{colKeys}{rgb}{0.6,0.15,0}
\definecolor{colIdentifier}{rgb}{0.7,0.1,0}
\definecolor{colComments}{cmyk}{0,0.3,0.99,0.25}
\definecolor{colString}{rgb}{0,0.5,0}

\lstset{
	framexleftmargin=5mm, 
	frame=shadowbox, 
	rulesepcolor=\color{shadow},
        float=hbp,
        basicstyle=\ttfamily\small,
        identifierstyle=\color{colIdentifier},
        keywordstyle=\color{colKeys},
        stringstyle=\color{colString},
        commentstyle=\color{colComments},
        columns=flexible,
        tabsize=4,
        extendedchars=true,
        showspaces=false,
        showstringspaces=false,
        numbers=left,
        numberstyle=\tiny,
        breaklines=true,
        backgroundcolor=\color{hellgelb},
        breakautoindent=true,
        captionpos=b
}
% Listados de c{\'o}digo fuente
\renewcommand{\lstlistlistingname}{{\'I}ndice de c{\'o}digo}
\renewcommand{\lstlistingname}{C{\'o}digo}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%para c{\'o}digo fuente%%%%%%%%%%%%
%1¬∫ Lenguaje seg√∫n la tabla de opciones.
%2¬∫ T{\'\i}tulo de la captura de c{\'o}digo
%3¬∫ Ruta del archivo
\newcommand{\codigofuente}[3]{
  \lstinputlisting[language=#1,caption={#2}]{#3} 
}


%%%%%%%%%%%%%%%%%%%%% BIBLIOGRAF{\'I}A%%%%%%%%%%%%%%%%%%%%%%%%%%

\usepackage{apacite}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CABECERAS DE P{\'A}GINAS %%%%%%%%%%%%%%%%%%%
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhead{} %Para limpiar las cabeceras de p{\'a}ginas

\fancyhead[LO]{\bfseries\slshape\nouppercase\leftmark}%Cabecera Izquierda p{\'a}ginas Impares Secci{\'o}n acutal (Left field Odd)
%\fancyhead[R]{\bfseries\slshape\nouppercase\leftmark}%Cabecera Derecha p{\'a}gina Pares Secci{\'o}n acutal (Right field Even)
\fancyfoot{}%Para limpiar los pies de p{\'a}ginas
\fancyhead[RO]{\bfseries\thepage}%N√∫meros de p{\'a}ginas en cabecera
\renewcommand{\headrulewidth}{0.4pt}%L{\'\i}nea de cabecera
\renewcommand{\footrulewidth}{0.0pt}%L{\'\i}nea de pie  de p{\'a}gina

\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}%Cabecera de cap{\'\i}tulo de la forma ``1. Introduccion''
%\renewcommand{\sectionmark}[1]{\markright{\thesection.\ #1}}%Si ‚ÄúSecci{\'o}n 2.2. Primeros pasos‚Äù es la secci{\'o}n acutal,este comando nos cambia a ``2.2. Primeros pasos'' 
\renewcommand{\sectionmark}[1]{\markright{\thechapter.\ #1}}%Si ‚ÄúSecci{\'o}n 2.2. Primeros pasos‚Äù es la secci{\'o}n acutal,este comando nos cambia a ``2.2. Primeros pasos'' 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\def\latex/{\protect\LaTeX{}}%script
\def\tex/{\TeX{}}%script
\newcommand\latexdos{\LaTeX~2.09~}%script

%%%%%%%%%%%%%%%%%%%%%%%%%%% Script para escribir un solo comando %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\bs}{\symbol{'134}}
\newcommand{\Cmd}[1]{\texttt{\def\{{\char`\{}\def\}{\char`\}}\bs#1}}%para insertar comando el letra diferente
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Script para crear un cuandro %%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{block}{\vspace{8pt}\begin{minipage}{\textwidth}}{\end{minipage}\vspace{8pt}}
\newenvironment{fblock}{\vspace{8pt}\begin{boxedminipage}{\textwidth}}{\end{boxedminipage}\vspace{8pt}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATOS DE PORTADA %%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\autores}[2]{\def\@autorUno{#1}\def\@autorDos{#2}}%%%% MACRO PARA PONER 2 AUTORES

\newcommand{\autor}[1]{\def\@autorUno{#1}}%variable autor
\newcommand{\@authorUno}{\ClassError{pclass}{Falta especificar \string\autor{}
\MessageBreak Usa el comando \string\autor{} en el preambulo 
\MessageBreakpara especificar tu nombre.
}}


\newcommand{\tipo}[1]{\def\@tipo{#1}}%Vble Grado o Master
\newcommand{\@tipo}{\ClassError{pclass}{Falta especificar \string\tipo{} 
\MessageBreak Usa el comando \string\tipo{} en el preambulo para 
\MessageBreak distinguir entre Grado o Master. 
}}

\newcommand{\titulacion}[1]{\def\@titulacion{#1}}%vble titulacion
\newcommand{\@titulacion}{\ClassError{pclass}{Falta especificar \string\titulacion{} 
\MessageBreak Usa el comando \string\titulacion{} en el preambulo para 
\MessageBreak especificar la titulacion que cursas. 
}}


\newcommand{\titulopro}[1]{\def\@titulopro{#1}}%vble titulo del proyecto
\newcommand{\@titulopro}{\ClassError{pclass}{Falta especificar \string\titulopro{} 
\MessageBreak Usa el comando \string\titulopro{} en el preambulo para 
\MessageBreak especificar el titulo de tu proyecto. 
}}
 

\newcommand{\tutor}[1]{\def\@tutor{#1}}%vble tutor
\newcommand{\@tutor}{\ClassError{pclass}{Falta especificar \string\tutor{} 
\MessageBreak Usa el comando \string\tutor{} en el preambulo para 
\MessageBreak especificar el nombre del tutor de tu proyecto. 
}}


\newcommand{\departamento}[1]{\def\@departamento{#1}}%vble departamento
\newcommand{\@departamento}{\ClassError{pclass}{Falta especificar \string\departamento{} 
\MessageBreak Usa el comando \string\departamento{} en el preambulo para 
\MessageBreak especificar el departamento al que pertenece tu proyecto. 
}}


\newcommand{\dia}[1]{\def\@diad{#1}}
\newcommand{\@diad}{\ClassError{pclass}{Falta especificar \string\dia{} 
\MessageBreak Usa el comando \string\dia{} en el preambulo para 
\MessageBreak especificar la fecha de presentacion de tu proyecto. 
}}
% Usa \cdpchapter para todos los capitulos que comienzen "a la derecha"
% Y no tienen numero , Agradecimientos p.e.
\newcommand{\cdpchapter}[1]{\chapter*{#1}}


\newcommand{\hacerportada}{

 \begin{titlepage}
 \centering
  \includegraphics[scale=2.5]{img/us}	

 \begin{center}\bf\sffamily
 
  {\normalsize ESCUELA T{\'E}CNICA SUPERIOR DE INGENIER{\'I}A INFORM{\'A}TICA}\\[1.5cm]
  {\large Proyecto fin de \@tipo{}}\\[2cm]
  {\large \@titulacion{}}\\[2cm]
  {\large \@titulopro{}}\\[1.75cm]
  {\large Realizado por}\\
  {\large \@autorUno{}}\\
  {\large \@autorDos{}}\\ %%%%% DESCOMENTAR ESTA LINEA PARA QUE APAREZCAN DOS AUTORES
  [1.5cm]
  {\large Dirigido por}\\
  {\large \@tutor}\\[1.5cm]
  {\large Departamento}\\
  {\large \@departamento}   
\end{center}
\vfill
 \begin{flushright}
  {\bf\sffamily\large Sevilla, {\large \@diad}}
 \end{flushright}

\end{titlepage}
{\pagestyle{empty}\cleardoublepage}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\addto\captionsspanish{\renewcommand{\tablename}{Cuadro}}
\addto\captionsspanish{\renewcommand{\listtablename}{{\'I}ndice de cuadros}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% El comando \figura nos permite insertar figuras
% comodamente, y utilizando siempre el mismo formato. 
% Los parametros son:
% 1 -> Porcentaje del ancho de pagina que ocupar
	% la figura (de 0 a 1)
% 2 --> Fichero de la imagen
% 3 --> Texto a pie de imagen
% 4 --> Etiqueta (label) para referencias
% 5 --> Opciones que queramos pasarle al \includegraphics
\newcommand{\figura}[5]{
  \begin{figure}[hbtp]
    \begin{center}
    \includegraphics[width=#1\textwidth,#5]{#2}%
    \caption{#3}
    \label{#4}
    \end{center}
  \end{figure}
}
%COMANDO PARA INSERTAR UNA CUADRO UTILIZANDO EL FORMATO:
%1---> especificar numero de columnas y su alineacion ejm: |r||c|c| r=right, c=center,l=left
%2---> especificar el caption o titulo del cuadro
%3---> label para hacer referencia a la tabla insertada
%4---> contenido de la tabla separando columnas con & y filas con \\

\newcommand{\cuadro}[4]{
		\begin{table}[hbtp]
			\centering
				\begin{tabular}{#1}
					\hline
						#4
					\hline
				\end{tabular}
			\caption{#2}
			\label{#3}
		\end{table}
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%DEFINICI√ìN ITEMIZE%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\labelitemi}{$\bullet$}%enumera con punto negro
\renewcommand{\labelitemii}{$\circ$}%enumera con c{\'\i}rculo 
\renewcommand{\labelitemiii}{$\ast$}%enumera con asterisco
\renewcommand{\labelitemiv}{$\star$}%enumera con estrella
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Redefinimos el formato de \chapter
%\titleformat{\chapter}[block]{\normalfont\sffamily\huge}{\bfseries\thechapter.}{.5em}{\filleft\bfseries\MakeUppercase}%[\titlerule\vspace{2ex}]
%\titlespacing*{\chapter}{0pt}{*4}{*1}

% Redefinimos el formato de \section
\titleformat{\section}[block]{\bfseries\rmfamily\Large}{\thesection --}{.5em}{\bfseries}
\titlespacing*{\section}{0pt}{*4}{*1}

 %Redefinimos el formato de \subsection
%\titleformat{\subsection}[block]{\normalfont\rmfamily\large}{\thesubsection}{.5em}{\bfseries}
%\titlespacing*{\subsection}{0pt}{*4}{*1}

% Redefinimos el formato de \subsubsection
%\titleformat{\subsubsection}[block]{\normalfont\sffamily\normalsize}{}{0pt}{$\,\blacksquare\;\;\;$\bfseries}
%\titlespacing*{\subsubsection}{0pt}{*4}{0.5ex}



%%%%%%%%%%%%%%%%%%%%%%%% TITLESEC DEFINICIONES %%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\examplesep}{%
  \begin{center}%
    \rule{4pt}{4pt}%
  \end{center}}


\titleformat{\chapter}[display]
{\sffamily\bfseries\Large}
{\filleft\MakeUppercase{\chaptertitlename} \Huge\thechapter}
{4ex}
{\titlerule
\vspace{2ex}
\filright}
[\vspace{2ex}
\titlerule]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFINICI√ìN DE PART EN LETRAS PALO %%%%%%%%%
\titlecontents{part}
[3pc]
{\addvspace{1.5pc}
\filcenter}
{
\sffamily\huge \thecontentslabel\\*[.2pc] 
\huge}
{\sffamily\huge}
{} 
[\addvspace{.5pc}]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DEFINICIONES PARA amsthm

\theoremstyle{plain}
\newtheorem{thm}{Teorema}[chapter]
\newtheorem{theorem}[thm]{Teorema}
\newtheorem{teorema}[thm]{Teorema}
\newtheorem{theoreml}[thm]{Theorem}

\newtheorem{lemma}[thm]{Lema}
\newtheorem{corollary}[thm]{Corolario}
\newtheorem{corolario}[thm]{Corolario}
\newtheorem{proposition}[thm]{Proposici{\'o}n}
\newtheorem{proposicion}[thm]{Proposici{\'o}n}
\newtheorem*{teoremasn}{Teorema}

\theoremstyle{definition}
\newtheorem{defn}{Definici{\'o}n}[chapter]
\newtheorem{definicion}[defn]{Definici{\'o}n}
\newtheorem{definition}[defn]{Definici{\'o}n}
\newtheorem{conjetura}[defn]{Conjetura}
\newtheorem{ejemplo}[defn]{Ejemplo}

\theoremstyle{remark} 
\newtheorem*{remark}{Remark}
\newtheorem*{note}{Nota} 
\newtheorem*{case}{Caso}
\newtheorem*{nota}{Nota} 
\newtheorem*{caso}{Caso}




%%%%%%%%%%%%%% para crear con \m{text}=<text> en italic %%%%
% Make active <...> produce italics surrounded by angle brackets
% (used in verbatim and \verb).
% << produces a less-than, and >> produces a greater-than.

\begingroup
  \catcode`\<=\active
  \catcode`\>=\active
  \gdef<{\@ifnextchar<\@lt\@meta}
  \gdef>{\@ifnextchar>\@gt\@gtr@err}
  \gdef\@meta#1>{\m{#1}}
  \gdef\@lt<{\char`\<}
  \gdef\@gt>{\char`\>}
\endgroup
\def\@gtr@err{%
   \ClassError{pclass}{%
      Isolated \protect>%
   }{%
      In this document class, \protect<...\protect>
      is used to indicate a parameter.\MessageBreak
      I've just found a \protect> on its own.
      Perhaps you meant to type \protect>\protect>?
   }%
}
\def\verbatim@nolig@list{\do\`\do\,\do\'\do\-}

\newcommand{\m}[1]{\mbox{$\langle$\it #1\/$\rangle$}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%conseguir manualmente una p{\'a}gina en blanco:

% \newpage{\pagestyle{empty}\cleardoublepage}