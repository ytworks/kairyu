"""Fast CPU tests for the fail-closed G2 A8 artifact verifier."""

from __future__ import annotations

import base64
import copy
import json
import lzma
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from bench import dp_scaling_g2_a8_bench as a8

_TRACE_ROWS_B85 = """
{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;t-
Rh2wea@h=?Yvx8pA9tu4J(aO|TFIeY1Cyb0dm7cYUYa1Vk18pk|NPQI~lB7O!|A^z8lkd!CVdO?23{C<c6E-
;b|WY<&bZ|PCDZS_>TPi+56(&dsLkNA^p$U{G6T(w5Xn?DLq=y|_OkdsB2<Af=3*LvX{3Lidq<J#y#62RyA3|S5?g8oFz`F}4mK*
Jz=0c9t#K$C$5G{%`0SOKBN_e4e;D15?Q%7fv$%gGVjRH^h<zwA|8lVFr@)lpY!5Gn;Khw@geLEyRPZMPa1=nG~P!%<CTGr%8d<+
o!%34AWgVO2nfZ_Ygt@f%zcleWTeB-
t9Ubi>VteRu|)5=<q~OFo(s=zz{3Pv$T5V;a9%zwzpT6jh(uPuA}EcS^(_m4mG9;?}zuNffyIxJtXazn&EmJHzYmAlg4@a|GSR3z
uD@I&>&z^DM7YH!oPSztHTO9!3PPFqVE`mQ^vaoeov+0Z40|IfXwjsg;;vBO9q026OJqq60?Zdc{2jp;9_W?7N%Xu%Jk}+7p*tQT
&fN66qo2p^yfhb|~W-5Y$rBmBw#bq!fd;Z)koriduLa2IHuvGEhN)Ras!SRA+cgo6O8-
LyZhpR;RUv<=R&>;Q00E=qH8GN5hl3&u!?q2mxxw_<_|qJ<n^Y!Ij@s=A4<70xl$KN#2{>zZo!!DWy&Y!|;uGbnA!nOm|@9-
Q0s_zl_D{@m+`dGq3A*0l#PG!uKXhVsu?LdECIhA&TQo>gENQB3o|>pn`s!BwVS9=>bPE3z~6V@iOWvvlekLSU9<CG9YPldn$5mb
kGn!Vo7$mTd}{824EPIPe*K>^j7iTTV)c{@2Vfg!cmZ1Lq0DZo*gR-IhmsF2?YH1z?rogN@tQlC9<AFW=VNYXZEy)n_M;=*{f8x3
JL#A0Eq6mTQvB=&jBD7bCDYmm;Ge`MkdBNb<V3qB3rv&@J}xAZmGY=I&vA*VV8oa7aHt7*Tz<1Nc0A-`YmRd-tT(al_*3zjJ1Xe(
1v!>PfjL{>Qk|^^zWA~Ef6b+gXOSR8TJ`6<@Et-|5&ehIx{vBuH8-hTF(Nh=xuOe0$k&K=#*L=t8;`DmS=x-
(8G4VxA^mancN8KuS&U^SLlf&Kb9@54nD~)*{&O=#(3Sb_L9sdPTszzYg`0vOyX6}&~|B1W^n^l3y;CqWsvvU|LETu6bIo+9zQRj
wHOo@laNcENHTf#ncn>A)b*an=SGMKBx?eoy#2ainmKL?m2LMA<6Wi5$lE34KJsNSq|tqQohOC0I0}36=QSg`LlfL}eG8nil?ZJV
A!BLVaM~&T&zqXHBChIQX&)QIBghi2jin?shZ?V4xLH*rX5*>NZe1`~PfEC=qwj)G9ay#rau+cI>yV>c>p|Q`WFc4#EiVA0VQzhx
e|5G?Yg}_<%LuJ;tfd4waN1s$29Gl*Zmy?b`t>pylIYMD@|^5gfeA;zm^Fi0Qp0thbbb7A>_Z}Z$@?UUW~FY$-
y@EpzJJwyEzY_Q;@fWQg|jlt$GPGdhTjEdzZU+R!9oae`fY?KQqxf1F~Sj=356#InFX-
x9pTog;KFYnaOJB2w!l6^dn?Oa8<0{2wLaFPC`VJNIQXsMW*-
diI*4DLuq+wLR=GLw5X)b<Zf~#A6A#8TIPTWSpH<Y=?Lz!!Uk+3F(1Jl+V=2`pMs$KR1}+j)SvJ1~*u|Yk(+YS<iHIwy>hF#_CvU
rV!6-3a$rl1q9&T|93WEkJ^axwQwKgbMi&J{2M9L;<KqNd~GL@9Jn%>!3Dmy!43^L-
}3P2#IecOcap4vbV?(xzTkAc74waV>)U7PYX=5t1dJw?mK&WF-7^ED<)tBdUlIsEsQ4WJ8%I^4B+P^r13MDroSDx64aVmqk-lmQ!
inkB5n>^CHQa3L~O<4zNyB0|^nXX{_z7AI1G59WiYQ9#(Q_F_Lz_14&DR=Y*d(RiBi=yV1fV<BJ!bM=qF{}NyrU~zEd{_jk1Di#A
uE<xO%(ezs=r<M|2Jrp3F(J+Jn9IvNhY83bKhMcv^_5*qKQan}GI&YjT|KS2lIDN)x=ndd9!ojCGw+#=1w@i=K^NS2R13$q)KYZJ
^ZAyshSCkpO<&i46TjlxFX_j&QzQEJAT&|?LbcfovzN9d!=npFAl-
X!tL|;M=M&Z+!VJ(DNT8FHt{4yf3zm*BezmGy;Tablc*Q6>S@vA_19j2%`;X11ar#(fK!8xD5WtTchyN~%GqI{Usq=Ko=?^I?yS$
gfi6skV^kVmqJs1y#hHv&nlZ7exEgSV4@=~37t&wG#B=^{9pK+|PhV}@1;dMA*fqF#JS+5se699KRV8Tr99$LYvBp<e~a`hLSw9K
+ocfcspKXU4SSuTLjFk=99GoLJAGC5Iwv>`gw2pR>EW0$yMThN?gU^{=#(aJiOwbc017qdelN#CVU$(@NSa9zDO0I`Z1^4#if*XH
J6Dv=bTnUQGgHxLWF`83e>tvpU}W1wcfs|DLVKZ3w|rMa)gXJnVxph8=&DO6FR<gW0ha6Fpo_-kK{%$Hg(l*Ovh)2;1qt*!n(If{
O?&0*?Z;0*Cw4r_WF+WYYEd^J$xSd{`8L#O@_~JVWQyoQw97S+o~`*g{^Dzu~@V$Zs}ezCobyBHR5ozv>(3HhS{Jx){sCP%5nU9w
AV@Qv|kT2>;$_n>2?!_Bn3dE}h2a_!yc)HnbZ<88@YzSE`x~S;Nt{W5>WE3Ztl|2cTS=Pn3{6r1zy~MBo%95i_%V*qm}NM{TDnNN
^Gp<e2;A_eF6S@GBjij735^F<1;V@%3P)i4#uKfm6~TjkSD0)r2gQ+HaLqw*3U%@{p8s2*>|nrCK$X(r5|6u4!e?ds05VOhPFWz1
-)z9DnO>H+7Ed0M*9-#)eCEsfD1<N6&ilTDbZ!lxcbMQ*}8<0L4;E`pT8XVRhjV0(XsxJIhV=&cONP9gTT<@V`%@-
i^+EjQP(g=8BXr){HMj2$WXU4xj4ec7c_=QLplkSbMS>!0k3coexlTd20#Kw<l}~FtE|afoXmqXiFi6{wBt;b;Ie72-
MvS_Cz~>#B_yT2IgCJ06M|Ux^HVEmHXQO4r0*{B1Lc3QBdx%*8M=d6FT8s45I$jUgj+<g2(Ap|3cz79c}GXm7i<uYN|?*HY*9v`|
t$&V}k9?b()n3{e%`{F}(ehMpKuMm7|!08|i{==rs6%DUA0~w?*T#Pt^*e2IY^qO3L;(k*3nIel6&ecgPMVg&_?CynY{OQ~h5LV7
Wrl^38w3wLf}CaH#-
}YpO)Adl#KgN+33*ao!YyPqAr*_ZXNh^jvjvs+rVt)eR;Ug69Kq`Iy@pJJ8iqk39Xr5yZirW+u=MiwGv(mqRyGb0OL2kAl<IVLU@
>c=mPDfqASfRdiwm&CS~gqU9yN9Go}xcOlG&5F)vR!DuBXn71&J!>L3g2)&N8U5@>nlhDptmXB420_~UVX!SM8zy4X?5q(1*^vi+
Avu9$?BpWj)6UP3i?lR1e98~^-D1RT1F!#p50`m>8_5J|62B~d3Ucm+J4kg1Oizq$C7HE-d-2o0Ih5;c-
H8;q`g406=TRg1qS1k%*64C(py_zhW?u-H6-ZtF-gM?2m{+Y-18zkN@|7M(~c_M<{VO-QbJ}D&#7<V9e<#ze1A4^c5l4*tH7on-
8aj;k~U;?G55+7Y&q}b;RVKgXkPujp!@O)tPIbby2@g}erXd=Lkhz8|Gt|_9H5LHGeV>8N3D3Nb3iETti<W>i&W?nR@;|AlrUH%E
|m*v;s_Yv(O^KQGB`A>f`p-
39t6%?(=bZua}CQjn>#VDmQLeynjP^VE{*{MSJL`_*=a{w;=5X^t$3fLeW7&2yn+SR3f*AT#7a!Owux{3Kk4vz#P1^joul^b@(Da
R{h$6~=i++VYV{ABDbp;!^M78c^?%B|0F#_C^m6KP~RLfBSCq*>8^0?iAk|2K-
p4hnDnFqtu)qu1RkL@kv2+1ky>J~?1uH+08^MTPD{uqAsivmEThEcbNLYl$jM;7G?0Ou33_vKkzAg5$+;1)(CDQ_=n<jg2n)eDhs
m0dk%&rT<~7Y(*z&8(s&wR%de;aEdeHw<ceC0wZ@)ZH*X{FDf(9Ug)R*CI==XCMdbkbeo-lbG5v^ZnWb=qv<Q=HAj852@N;bAn55
11K5=WN_`fMLuUi911!ay6i+AIKZsNs*EIHhi~Y)F-
3@L%Zuh=18^*g}f<a(QCKypg7S4k1P5<ntOjg^=h}tT;7k3r}c8NSi9}EXi1yt|9k(P`U!p@=#4@vjdE1t$K_#Hk72@QXCvf#%mn
L7i%O{sI<vQJ=wpba&{Av>5Wn5`NafdeOuCM%;DTGiiF8#4;Dm=br7Ku9X`roF3#Gacnb;256i^jSs*A6*&p2stAvExWHW@H?QGq
SGVW5)sXfcooLQ+&HDG9V(Dap(>$xK))tjN3?JgepeNTWpbK$3?&*89)rH8?3ufr{F(DLXiuP7IrLUOPX9#poE%$Bi9!bEcljQ!?
+Ye%3|D~aQRW3a=j@6(VeaS<lrO-Ni#qnUL-En`mOQJ*`c5LI!%;YH(W{0j$uvxWSxnX5Wnpuoekl2z&MLf;JnMH|9Wh4Pgc;PJ6
IuWNgJJDH)FENWMXxc262qbXS{4&vW@`Lpq;b4AJh4Rx_7SOs$~p~yV7*s$wd8@a^Pp@0J#>Ey<^a+#2-
A^UxhP)b{I?pH{tsEwx>tw}_b6j6+XooH2%p6(nX>nFZVnN94Ak*WQJPgx=TwLSa%us2TDfC$$o<&=&h}GrCr6hSLhVHCyZ~elZn
yC+iZ&*Cd+!uvfUQT<c7G|s4J3RKjR!+x6y}P{?^0oP#MsdbL3t62gOo4vtpcGVTy$UMWJUR<C{v;f)AIhX*{}3nvt1#IpJRF+8M
G42iVcYj?zf>-+S5DC?;LXyZ|!iG!u?fKy3fu#<m6$>3--DMNBna&#C?+%1svhlm}(|K^2C%{1HWu>=Gv8&<%fz}2F}s095YLw8n
3wmSd!X8LW1{CLOTNaRyy1rNO2E%qrNBS4ah=#9U{PE??3w8PnNnXD$tDI7bGHVL$<I#|LaVx1L}xHYRsXuCm&1Lfv#Fal>7#bRU
=)CRDXCKx#(BLSQcnOSt8=MqMYEggq|V&>5D2#{c5pleGbIsaNE%T2PBuLPw&&$n_)=4*qW0@EaCrIXCT@Dj?#+yI=N#e#4&`k2?
;uf3z6v1vaAoPhF;qm#mY_%hZ*EwzU1<-fG0bO(q*H!gtI0LlCzO=jz?9(8R;YCkcMv8J?88D5Ty&fd8|_&pCQwfU-
k)yAbI3ugtrA#mPf^oQ`s#kMJG$fmqL<5*(SHN`5P3|HzyjgSs-Sj-XrRk<-F`tGq$$<QChK#@X6K~qN94Vn9tMjd-Ziym%Uo_$B
1EuQ{YeB4XI3R_2R7vMpOc6^Btw;P-gK-o(99KPbw42;@8MUqF1yLqG8M-DlEtaTZ-90)tikyd&bu`y`M^Ei$ZO16ON0z>^n;%-
>mh2kp!3DaS?8D$knhkeyH?2P6qlB?G7HiTb^6+j+B2a{sKbli2nwHAuB2q+7(*HHC$1cijIM~tkHORo*X#|IStDF%gSUBl=jj^h
gcXpDOA{zRN~)CM%%IyEmXUVT=&G4osfGG)qoXox&YCKQI^#}ZNv`|56YaW|DX9goNS&7ms3r-
t&s(jGtnpledKb6m`HMs$Ed=>HQiq{$uOz-5b(Hun6p;Y3dv?7W4Aa4t)y$6{nG_Q*f~X$eISqt#k81lyz`MAA@-
c*CH(6itC5bggfgwVh3xFE&9O)FwfjBXH#RU^%dx^w%nDR1gz77MVY&xWBtOXSOZ0)FekrVqzh-uuOJ<A5D1li_nPP9z5mTPOiVp
ve-
N$guj~Au!sG}@c&W7>4TG^y5>^ef(y!+&!k49%wl~TmF+>fs}V(O@&Hrd*l#2Nd6X^SjIna~gd1#(MIQBD=r7eg_G=j|XF)Jc$FJ
BVIcQ+uWqlQHB{d~9_;WL3VhgJN6f?4U0`Q<D?x$(^R8Hx&k5W(Tn#J&uR=F$<oi5pk3O{jj2-
BAASq6XD|qhFRM~LKXWgc$P4q3R~vvn6<`~Woi22grr(Seh<x?s1y(ru6)Z&X`cO4^Q$jWnqAl3XoH-
@jfiS^RC&xT_?bsC<+@QeX?U*BUI^K9;WKb}^kGP4zY0DU>UEY}mToT&KU?I)PH`AE*_;8NprXtjef~6Gp)Nk66%yGH?D+Ir0L8)
#ri_elqhB2+?vzccpX;K$Lyz$ySnxLT`s<%d9-
`zMW{LEHZ8_;<E1==)^&d&*JAgdX3OQ<7+_XpVS%!JCVVaE)w7YndPQW0$(eP9p4dS~(n~=FqDcJHkKf7o^wBKBig0uBTXz0uaMo
mkIT47u3Zwl>F_%=qaq!H!zz!OcJqeRWGZ!K4yj1CD(hGW3W4~)@YiK$(|p4Y15V6)y*v;T^Ipv&*>MVL*WtyJagNbnJmIi2_&qR
EE-cl4b-)qnv1CTE<dh7McX^Lc3ME~-
Wm=~rTF$jWOZ+JsWpZtzR;U2qx+`v>k;+Tx9hiA86V()G^U)1dywJOyUF<l2+kXA9;ccb%F8dJBF^*?LJy^h@N(U?WDZ)@@i-
X%{%>7vdx}-b6K>RVr7QwJL%86+w}9_g{nf14XLSgIIpi(XqM`o!(yLRjW0!T`&a5JUi}?8&3oXeZ@QWn{>-ms#%1s55VeJo-
4IbB69-LhB9gZJAgs%<Lq1COiXc-
X0^2&m@akh37OA~_2`*`(d_uh9{Zn>2++He`2ckfnf<df!V6SJq*2X?5ogfb)Gg|<KJxMg^J0*#bBC|u&3D>os(X}3*+Oc#E94cy
sS?gN+Ud$Pt4hQAw+;cO#xyKjaq2yPy%3i8FHc*QpdxR}MMnUqY&0|7Ne{WF%$JhQQEAkN>{RA^q%T-$SI>&2Tux~d=>_-
rH~%!BJ)F5OYN$Uh7Xb`OMge6Vj&9S(S#X4fA*@zJ#{Oc8J}r`V()+Bm1ZndQ52lEo$2EX7Q9)Lyc$t(jE?OqfCujYvi4@pnR(eD
3S>hp$g4{zested>eoDDNP2(rJaL%K5W|flI#-Zb3P`DKTOpa-v(7PJGEKR>$w9N;*gur?wC$333EZZPj$PhdJ9UV6?-)0E-
`tqhIV?LL5Uy_eQbs(=!YZcNsw0r2`m`Z!>qII%#@0+z-
0r%pM2a7YKNhsc_4j3cOI6BC(pK;e%J3LNzqX3{kcQMpI#qa_fWmHW8>F*hGOik2n@w=YIk$3F;BvVF*u6cGo_thxFMx*9az==7c
iJePkC%hAvA4MPGy65?8Uo7Q2`EPvj4rmC2$FkF?s}q5k&&(}~S^GTo$JDw7O*doFb<;2PhxkDt)j;gXu+P+h(xkY`@}PNH2-
4AIxYZbj&2aNoMxe+^q@0(x%a3QWJ@bh}N9^!NcLfr)Mwp2(h2yO{Weiy2u~&^?BwMtGnKY+g0VTVoIXL40+&V{LTWfL2Ohye4h2
q)gL?E3ccx1gKZ~*5TPf0ZUwz;>ia2(Fd0+mY<({p(TRmizRwyS%MC`1lm$X#AzumY;H(qKn}s780uo+~hz`rgk-
R$J__P^e!E(eWdC|BTy0rS4+hlB>OWu2e5z{nL0|$XiD|VS0r^M}jU-ZiI@KnSoA!WZrFY;V_h_W)RQWm=@vQy(HLK*1TX(_SWG~
Kin3%RPQLPPJrd38PlFgn;5rrz<Nd(kuKU7?xNmMBZ;$m38pg|$}}0%u!V}`UINC#(Vw~wy@#)*GCvV=_w`8AUzwmySi>6>?HQXX
Jd>Zx{1UYP5R!03_@D3As`PV3-Ug-
T)Na2i8l8?<kwz38Czn&|?N&KiW49;d@}v84y@)jQ(T15qF4_0Eui2H`T%bd?$w(JP)Z^RWdURaJ6WFFfLvD|LDGi9Q8yKsY_sQo
hPOp?SqFdTbrJOe)Td*?t+UCcL(C}tswSYNqf7!&bS@;MqI%oQFpeUd`%6^q7Te`wo3q-
F75AmM6V8F$JNnWPwsLBQ4^N~2YCEa~o@u3uF*RnatKKl#D<lIy^I!>z8_XP9Q#t*B6)r5St&$Dyc4o6ynnSma=<^r<7+}BMrF-
<0qvYU^^(EKWqfaixcDGKWVOkR=tr<@-c4(A*wV8Aq#S-r>-
xoG!P9qN#Vu;b?mEE;wUU4rGxg2~<Mc9o(EpMiqh_b_a3gG!IFfOU)TZ9wm4i4YcZj^fPTuSjW+z_|ZmtC@lQNyl>(JcowJ9S5VB
Qw-MdK===I4<X=u``a*IHtPb;tu@O$Gl2Th+9~vY7X`o%dRTS3%N*i~b}h%(`r1=RJn0!Gbr})qDJeUps)o@zjYa2Tb?}!RZXr-m
naNdpI}xiW0Nz+&Km;&(u8bYjoK_>)8^=GsRZm0vg$j04NW$F$JF1~s&Jq{2@S<*lkl#pgu%ny~dwisvT)EEqvq`Ca#c9jFBa;03
IYX*<?~DV?90|#q_glBH-
~u$Iy8uuO_+}?SAUGTi&E@nVx8C1{T=4o@f1fa~o9_mjyod+CDkhJRyHoKx<I2sEIznSdoUcbqPQF@;Xnp@S&0Rl2UP={X6k4_3L
Nus<FCR?1mOTul+G>KhF(wcV5H($afltqcn9~@lA#0|lTROQGb3ePGs9!dvjZ+>mwgkSg<NB}BUtt0X{KBdA)58@Mx%6x%62b&2G
S&>4SiX_;$kYs_BBt<xM#xM2zBS;uJBRM@i%!*qjVqK93?kd`5L_M<)|2>n^PE3;aWRsEaU{`x+O&P-
N=y%S5+DMyno_(mm2eUw@9ao2)=}K0w+0_#(&ch=P4&cj5#rfFfOif=!rN8uoU26}NyA9u(Fc9v@;c|YaF2iF$VE#P29OX<)2O7;
K+aH`P^H|{vyNZN@vpH*4pw{MBq|RWtb6%Dv1Eqa)kaw6M=54*ct-lw<wYih1!SNYPl>_`67<gv6T0XowR@%(<X_Sj>P~e-
x6|Q4XG92e-
0#MYuJ{`b#8$nmU69!n(!aCDMr4*){&p&MpWjH1z)bREFd%524)>5TMyz{9KV=fxH22>(FT4VdT(A#$VLDeB%bZ(X2UF!&+%hANe
AwsiBqJql;H%t<|8ub6<(KMqax}Oy!f|}n9g51lU(1Rd641>@G&aF-0}>;<ej#tMhcm9()8-
^ceK8bW`xT2CB^fA~9M*>eM~t<7X+M@WVYwY{AGvqq{jwDIx~jRoB97zUQx|RzhVEys2E){82zy~>A{e|!m0_cd85%K6LcwkO_p%
Hx&2&s=HrPqMjw|GU3IgrTH1}gCuE`fv+8?97(fFzXDS-S%0E?-
fwQk+HLHNpW6XIeq^XpE%*6O7IZZs0>JNQ$IQJ`z8R3$MUy|^RNTJ~P{^YtUN82@FUcWqx$SpqBIRrtiJe(JW>sAG!vr%dB84Q>x
Iv_$}TM0P>rJ7&p0>nSUbhe<7OvMkus4jBiWVBTdPe;<=f=?x#bThf5u1K92kx}xLp?2|(FYA<H+>P|L|<7g(REE`%e7!BMO*u#~
*aq{WGZrsG28L0?LGLZ^$;y1&S-2-N<-T8VQAcx2nbKEwilX(0{7s3TALgB9Zg>t|>zw4%oUuEdZ$IcWZ3(&&7dc#NEG3nWK11-
yg;}lwl0(SG2ddZ2iDD)fOJ4@UeK;w+%d?+yqsJ?qj-}_C?0B@-^ZU^pFoZ-
@QIi}Bp1rediZxTHyvXuqn8w%4#9?KKf@qxfEZwaQ=0MCbP2MT(~>HI&aUhJtb>D^Oe)i;@Vneg5*eP2L*Qj{KM6Sj??tL7}2N{q
l8ui7fP?thjYRKLIPJra3mO|4uruTr;<kdP~OpP(s<TEZu!#=w7-
7x4q}PUSUQJqTCO$@TKjJzx|%umxIHiiR2Ix=J8EanC;2TpruE&~M}dgZzT(eZ#+tQ$X~V-g1aq-
A>^5B<S3YBvlwZ<Hx9oZRUf_xtGcw*IgD(A@#`?3+Hz7k8iC{yl^Bv=DKeej<Z%<rkRbcZx&VuR27!BowwJMHcnJGaWxY<CoHv;$
l?(yEPji6?oXXQ`jBb2VE|(mq=Vj7Qb@#&x$CWtM6gVW-6C4o-
1$19)tO?oT(dYq5IBT!8oF{&ItSDal0xPGKcZv{xIzRj>UUBU5{TvkA7{q1)><`z4+Do~G5c@sJ3-
p@1T>Uo`?Myui~NG6WQ~~bPbJ%?!^(?*bm(m#i9F+6u8J=1DGfY}trLk0&5eciy%b|iz`oE_xF<k)%x=#9BkQvc2>iO~ikeF%N-
?Av9{I!YBe{Tu#Q<6|LAfwB*oGefL6l0%%uF=WVN7(U5&Yj3P>OkwYTNL0FfY@`rJRA#UR0n;RY=UJ<J#YcnDr#SFkVfGfV^rXqq
P^_=};`ixM(V7mx&%^L>w4u_=)uL*atSfE=y|bUQWH>^wgM1JuH|oxjQ5Su__XpYhSf!z6Ypat_G1(P=d}N^_OxKUNTUk7n2H)U=
vRd4TFA`HJ-<9`o{12q0n{*_SA9}0dT`_iRZX#g?w*w7;+Y9QYA~-d%A)Df5&2^uWa_{j+^tO+Zj29Q^Q>N^G`OEE-Nnz6ulwURV
G}LQDxONUB0vWR)A!%h^q0{9nvnkD`Ux}3vDQ<=kQsJVi?afOx8s1kuYA>f8I7Wj<8sn^0w&dBMtPLIa?{Tvg;~mr}df`zjGOdSR
FH{yh6>G4DO<{^jhATeHcAXqJ(GwK>=w>KKi#erX5v4bDQ`!&F#ifKML|WT3U1(fsy^ujj%YQd=ustIsWu}duWYw;Ab>;hNLUQj}
evuAN!r=@fRdlh#?W<+uGwA3e$ef${qnVE5!#0IJlIMVj+8e=^QK40jiwy+`-
P@00#y~y238a7wD3M+7MQqzId$KSc}}Vx{B8$CJ4X6)Ar{D^r?PvQsiCtsx6XX>}uvj$L1>*xu@968{>Lh(L<zI<l_%Ie}p6#N~o
AB1GHs_D*x^Zl^5rhGRggI$;vgx70~kBUo^gSf^>RN+q8`hkD|lK!w7nt`k-
k1A=Wh+SMHYrc*^76aOQ=b(^aUqIiH#=I^C)}fxJ=$EEfu2Lrd9G3QJB7?|KZ{H?6>QTiGl{JN4S^EvBk@gjMCFS7}IVLWBOiY2g
amk_S+q?td@hSDVBQJlu~po`R$@0K#jVpZlJzJx2sjLEgQj<Z80W8$}+C;tD}@W2%cRpBr`iv$auqOyt+Pivmk#CpHtyL=iY*x6x
2yRbxdw1F=+=S$iU$s%z+DlOC9oHz|%^s1A}O(EiIu+}WNjR1IdnW>eq&@UXJZ1>k{p3;~f#(3uP`QlF|9f%@EQ(NFBiN3WTX%b(
V;LKBtQae?jJY_m$|6M(8Nd)t0KhM}35{&yUK^6v!8iW{!<g6r0-
60bgZrgbh2L0AfOYWP)d<FB1GfUfz)>5q71whyBxN)LcElNPkq;XsSb!XEFnwn&|zWFR38`n6grliannI#6?KFPn+9`(QC{qZ26K
7CmzTzz#8ir$WW9nNe)0tZU>S)21HMw)I(LD8In9{YsF}a(1-T*c1c;X;LC>s!7nQ<I5JHZWb-{=OP{f$auum$4;A4_-
U=k<09_adVu$CiYFYf?=w(HTbI6c;WERA!s%)nPU&NwbBNoW2x#w87y=PumRFC2`k{OfIJzUbAq%zT3A*WISfWUZ>y1aB$F)lWaV
YFfJN90!uXBp_pGS}7*}Wdgk^z|SBTWz2*0;AF!qxB|5bgmuR!SMIBRZ@%Oh5KJ1*HYN@7f1>ik+#L%zkd9@|u-f)6tjd-ce0P5z
R@?D{HuKrz=6rWuo;SrHN*3sV7sXhsFAn9Fpx`XrvL`M!Ui(fAS~V7vv(3C#AEs39a=gf$giAPosQPt(yp{M=rsR4`4wgNN5Ttv3
*%9it(={&`eoyf1~wc-ZVY?!g1$d5FPK$lS|N`-
5Ir=d6~a3C5@t*o|ZhBH3Zmx+Z=5WYYG{m)|Bxp3@X_oS{Bs@s32db5_~I<jo**;Fh7V0hup!Ear%I)^setXT%?xZ#&-
?J`yZ{0(1({0&tHKzShT$l)j;&x$%}#;8ERREl{kqIk9TPlF1;F2^inpv3VvLpEYZBFm-Jy~7wU+#4ZzsbSGKDrNC*9OM(&0Mmxh
$<zlxw!)c2M!h3!fLCUa{l?iJcP_}!{``Q$OF9~KfegdJz;=qx@X?A>Zqysr~4A|V(cq91|UE&>v@!EW{3$Yl^Vj^;+gd*aq?rLR
&<wLA6oA=huZy_Aqo1g37LzEehB6XdU?>l4Cscb;-
wTW4hXd9)LQw{|;6L#IvwB+>a?vFxfw!Ew!qw>EzTLUvxqhv6>d&idKd70TXkK~EkFX7{Okj^rVm4i?`3jvlILFojAp>1{?3;vgv
Usr(3OlcTy)ZfXOlEl>w355dFFH#kTgfI_;2x&{5%pe=~US*C1wx@;3ujp(}ZXez$1)WuHjCLrkB;hHR#C>%CcN>B2e0IfJY@0*t
kJ4M?`jXXetw^#V6mVUWVfiCji-S**WjpYWWPUwkJuE<P+@f-
lFJJuS5NFp(yoA8@90J@4ES2L#X$<gDtYsoWDwT$wOdj$VjXmSyI*XeDvxJimVS<I<7@@vBWv$1^mQhx$u4>j~B6iuw^pD*t8rr5
*4zUcJLBfev{n=#&`cE+F8qc%yLs+m6xMymH=S~UbOTL)2VELE~0k(7G+1)43KJSb%ieCieS-
6$W{_jBTnacNAZc3wp`XCLg^^j&|_!x1hsfR5!%*IUi;p<PLaJUt^PclqHD_&e9qFkn4QUAW-
7of?AHdE@{#i_7)^1}sTN@TS!ZK%ZqO^1_3!^xV*#r7cdDBE?IahNXJEI~ruGKvd|&?%<AF>YFmq^_UZEom25iTdLM0HS2(A2UuX
t;LD8Vf+?9iAQsguB*PklCnPU+yPDEpWBYp4&<^#AkG`NWaAva%NK##3g&Q`h*UC!=Y4`W6%mvqB@rT^&?Auxvg&|l0!K&U=PRPH
BM$2YdNBQmMlKezcngEgV-6d8gJWSbn+kE_3I$c$L^%nq-<QyFZ2bZ@=pXyPPewHt`U8X9-
Q$|=hRcWw1SS4AIqp|(ykQZh8(~sk=trVjnlpBO;o@CZtwf^3X8*pljD^Yeby$)au<l%;?r4`jMGAKcG43t+>iBXFyigE*%&ajXJ
n3ndI$a#HyyIful&|R_$G!^9b3{*vG<K>BCUs&vSHG{plUH`1stOB#}UH#?aS%zzkZV8Kg|1hX^?w2o3F47+oI6AS8OwM7(Unrc5
?yLY##wy5Fk+(w<B^;pxQ)EQS0Tj30fc}ct*@EW2lVS?QnrkzsH7pbWq`hqoYUHym>xWnor|pFnzJz}2pJfT@gMb$hqOdteKTj}5
FOEH#r`7miT6eJHP9AZm_iI8CNF+8h8gbs4X=Q9q2*F3KmeQ)H69O`G{)4j){RE(fRzQr+B|$v%Esm(@nP!CO=uz?}2%+2`wM2IX
Rn{{3C&X)+QPi}FqG^NPN)R=@#-#TIFna5|P3=(*x4l~*yOf!kjB)K-
VMVm95aZW#zY~xwj3>*dP3$Ixj%3`p2c2vwkZ2vlxbZ!4A~|gI&%%2vKZ-
=*II0b91x7v(CO#4=sy%UdI)WhDb@`VJ<oNpHKdd^caqHpBCH@jo8(<QIn8>=)6JJ4$ujtZaNU;z3;H%m4k>`FF(s3UvGaQLvXrA
W$xg8122|K$)V^siuO=)oRXssha&boF@e?^`9{#;x*{Y-!S^T1-#AQ0hyN*V^c%?BW8gM0B7%@(M`xr%(=U$$?5%9UxxWX9z}Mg-
eu#v+v4MAs`%-
I}j$|55m>dI2vca7JR(<rHPWebMbI9Qu^l1PN;1i_3A1!8A13z*Kd}|H<v3{TkDzstT2nFf)q`*6j+kReo6aLjx79&TDqn%w%_2z
Q6~(fcb}i7(stUAFlT6=Oh<N#{@LmYA|Z`t?l*Ly;!$cQx|U!-
l$z(lwjF+mY&xOoZZ`1k5CarCmIvE9@gF+L4U%{A$BFypOpg0uKOo((5^lQy)n|PL1p_mbVkPqolq>_^fPq#jV|8iyWo7T9PsXE8
dXX>#+=FY<zbyfwIe6xIjbN4eR2?F6-A~MOnd*`0~#G5)_-a<h6MP1@o{aH6j_^4Ezil=M);;;@n)muvy9}1K-
skVF%O9pRyx+2M&^2r>mQd@G*eT*ZBJjgv*pVhcc_z_6B!9IoT0oj`DI9}h<193m<6H^#-qh2+cdLwN-
(w*?bHQ}ppt?0Ippd=5A$-<@4yBm^V>d)uf7^vdvUhE*rEEec~^Lxcd4-
^!RyPkre6yeR25ae;9#QSk^6y?E^^WA+kgH;i7PqrWI3UDr<797C<Za(!9-
Bo!w|T)yWqJ?K+1*~3R~5L&|haX9p!Q5Cxxq3#&n(T&9;g}%oybFHZe9xyaBz=lxwjfq{ys2b2KSOi-%{G;1GKd8+(zC0-
yy6@WJ>IX!uA-vO*OfXSp;0@7EWNFLCT&Sz^RuYCa=&NK!fk@1)+?@V%zy=E&2+>Rb-+rK3495S=wMYOd-pAmX{L>~HWk)jji0la
*tWU2<OwGDHFm-?0K=kxF`s0E5(0#*`RAMw66-
^u>iUDQg?fIq*_ZU4^6jo8x3Gs<HOmQdne*BBVT5$hV<w`|`Cw=<6qj>gytR3^T&7FYDEniHT6pGW{=xX6#yC1r6?wJFB!n($heJ
wQ7R28Fu>O;&JU?_uK4Ll>tUH_2(4<2Mnl7SD|Pk6hUDfA533zO)cg@*>`Y6pn2{5voXe!xCG1ir3fCtuM@tTs3Kragtw|o&=OGP
(8#|i|FcL<Z~wf9*{Uh|bp4?*03NDRHy+)|KH*lzffW2+wMECi(466rL_zBeYNM0mIwe@CA3C}-
Dq$VuYJ6v@F)5SjMhIGYx{rRSqY*FE-DSu@iz6qUI#WwC@?mN0k@MD_#Tg8qb6nR`@}c;J&;`+7ttk!^*PNG<4#3~!F-
{E>Mif=Mu9{SY!df}W-@LR)B|4}crcm_J+z24};#K&wUCEFeZ^aw00!hd{$XD;4cxA7wNX*atXfs!3yX_LV-Vf>_Remn5t)eF36u
G(;ms~yA@QwAKz8ZiPKws)VnqLLuY|p?O!%4`jwt_0|#Mj(sg~bi;i@ye6hq9o^pajidlNh<=6q80<bI7B{<?797H|0214-
M$Sh@jgJpLtj&Gp*{nj30SutUrSs*N1-
xjEMsKAv@H974B{6aE+Ao>r<WUVBfOoqOt2T`I0~yI)M$xRggst1mZfxcM3$c{>E6L%iG-DwQCF2W$lB`E0p3oZi7Oy71$Tjc7Q^
cp46YCAhCF+A+D-
HQMd{&U;@nY<FFxU9}+8rShHS)W)Aa?A1y43756$3wjddpgMHv&N)9c3VGS;mP!dHR^LO#tduQF_1?>nWwL?!b`Cra}=J?CA;pAb
EwvHqnI%KSS2<v6vKD?=V&Rs@sABod#uTf#DE(rzgH<y*Opu6t{pEpha?-
T#lt57?g+At7ThSpm)aOm#3J$RiO)Qi?VO}j@rWl|j*9G<d62jPA%Da|@;0yT=kIM<Wznno!-L-
(4nMa`!jSI6Nv@Hrf(SHC}D^<RTD^&TFx4j9i&6JGEZDJeo)BagRwrxkVm#DV?1Xe#Gg>MF#Blv4uFn<|atw?_}8_8p8r4zR-
%EZ5_3cD^A%%UOxLY9Ey-YsS#=OkZ@Tjbb5gU*6+KtDMty(3<@S@rGh$^3QXSCHToJYFoehu`JBJQTZnnovfgjiNokK$Cy_6`G6*
M*alBkquQ-<83F!VpWi7cX;gXvs6?h9TQ4%vM&_YKP67PHVC=aQa6<F7iQA-oJs^tMA)pBnr1b_<^$U+`<GTbwD&Xe_#u`2Pgm#q
`QUtBpm!%B;hi@*23j8W5OQ}1FXaQ?W&k~%zW0njz9c~a*I3Wuc3nbZWDST}eaP+VeB3FbX1J8efCWT$$H%?reX3)<kpW8E_B-
7CNjE|H{FA?^dD`P*mE>W8zw;$6OtUCQd(~>cAm)V;f|E!fy?Vz$hb^ip1(Tgk`hyv8Jb7Dx8xf=}yNYp{oA)9h0$Tn4WY3fq;Hh
lg~ou{xQcBX4ubgSlOlif`fI(}fDe4IjlDZBqXE8G!#^-|Q}=u)~Rt2k%JVEA`yeace#FoOS&-
1XZAS{j_P6Q5m}nMw?g>C+ts9Qhn+)pug&RPlONS^*&C3<BFt*YFTdcjD>8UGLH;9~>b724MzmaO)4{qp01oY;Hv!;|4l{n_&Oy^
h15Y_urQ!OUZYB5>$imiov*dXbWbZc4DUS*3VeMc<8jPz3rxqJ25iPKs{sjsyRCnP4J!R`==9`w4-
PN6lbE4r2+VsP*=eH?`jgNl&iQ+&LuzZ*bfnZf9X`)RRI4G0Etu&D3|!;VRFQV3;DgMF!q<&t=)$lh$YuTNT+D=a4#XK(z`^>voP
?DAM24F+$?7jtx^*A9b{F81EMbRn}Qz*>`X~<7?O7ji)uAtY(tsbcjVT)Q|6MY=`JW(ntY6OjaSy=VMHfbMjT)-
=|lX<eKjk?>@&@MTlaTkLglH!mGfs)oOZ=%AO62(Tc%HKB(TQciDlKAuBNqgeC@n(4GDof5JlLr%ND<Qx~lTXv{Lp#9%F>9AB?Ss
@R>u9Cg|P57;Z<p))jh!KtDQ9*AHtH2Ab}lV~jhIP`$sx(Yn?4!mG@ik*x_%Q7WIw3_VH@1k`mbDm73H%C%hF!;Kb>uNWlVW(LWm
&@2I~@!1B$A!7{5O?Y8NfjObK7hYn$ch(HDk1Cld0l64sQWfXCuuf&SSz=n-%E-|fEPVMn`BzfQZog@6p?DY47%rc-
4yVuTLe0&W!c*HvTuPJ}A0s*opz!(A2SAq8R?^ugC&UO5BG(fNSAio%RC(;hO+Qzk3Xco5{<%bmMR`3)%>~|)zTdeWuQB?j<-
AR0S?PUqQ{Pjh8#PLQ+Ya~gJ88w|xQS?sowm@Z$jJq3)lhly5SR5v)|jT0BGB($sO)pZa#K-
tpf7nQVk0z$ccWF7h<^34&E}?I{3|toJugGqbbcm1vYk3#-B9?(jXYQb^;`uf0p|LcZ_UJ!sxBi27-
clQ%9H6%BpW@vQ?&2W+~JV>E;7U1{wow9RdwcA&vDjyeM8ED^mX>H>f{g-QzF_Iz!mS^*Rxf};9-
3dWiD*52^Io4<*kNIxaHV{(Q94pZK<`v_a6*=3=Bx>W(D2=dBA~h_fgm6ySl&<i1{-&4=`5bB-
Cz3E&pHe#vG+L*{%FS8jH?aZ{ojrWE%Vf-vM+R&zZnV{YnS@&$a$DA}i~*k>WF$M^*3E_mqDm_zjc?7@lS=#PV(U-
mT4&?5((f4gYqidu8|09X@1z@scK+r;(mA(Bg>J<BMlEamj#I+aJ1yiTgZ)Yn}u8yji}Dlxw=oIZy~D-kI-
_8<<SV01R41LDvPd(fbk5eB$t<%CG-
I#@r9;w__%e1uUbxvXItWP9uu;nk9cyi6Hjwhi~(^Ie!cb6zU}D=g8W8O9wOHu<Os(u|=bHg82{7rz5xW=5UaLejiy<Z+Jp<r#CU
$HaR-vsm_cF1_P=DF`Wp^*jAX<`fLyE&8wQ1dXH=b{*^!2y!dM%sFhUm!*9ZZ6|2=NZo`ktO}~m1_(V%q9)Wf6D_E2U!-
K4qW=A&6<Lw>F?1;TT#Wqo8V?J;#qm!pb!C<FYGGT}N)04kFEsMyfKjsB>2P3JhY{^RFppdwf$0SVLtXiNy!or&ng1Lo?e$6=Dzk
k}$2Tmw&D#h9g;rr+4`E5OK?p#?}Pvsy5jwM8M`F(WP8(p`66B>{-
muOk$=)f!ks>uW~k!TalB^noXA1Jv8OxZ80=I}b}N!PXzuN1C6>Z)#Vq0+-
_KsdkoccD5ZChx9hl3czS*=6Mdvrbx~0FnO2aF#y#?f~^J*(?}jj;^KP&s(($Wyrh!Ik><lQ8FcfN`XA#njS%7KuR+G-
`Nd#Ok{r%9^3%-
j>~gj*K=5W`<@w&E3I4piYds3c1Kx_?#^VLP3Kq>7(jQ7HcsiLEqKMw^JH!V{gWiXOn)D;&2x5eJKEF*+E~+Wm>sC&i{3@D(i$GL
V%#LU%UQ&guPtWH`&$SeHFVSE?Vp0^hmHDQxOX_U7nlq@M-;t|8AkS5reVh}*gqLfV24XCqV*r3DC_Z3-
f2AGZfd=iRS|L=B?!h;xi%EEnU_t1R<)x3+7g%l*dA!XxrKEljBAa1$zriNSxr(xVMkU}tR_S5xmD&k=8%JLCY;CT8Pqgea*tW<-
Cf{jL7W%f+ecg>BFY{!7WlA!le6_IXyR(tJK6R2EH82=dUvn^8fx!j^?wwct33K7qrp{B2&29d$~^D^>Gt&r5j1>bbLsaKBQFWhA
{8iU;URCBqNY{R^A9HL7r23|Pk>%&kRW0H(DIGf8cBY8!pOG0-
pR6`>gk*tJ5jC{bgagSsHp45csgZ}wk;2_#Ap1e(k?>;pJhUN8k3X8HRFfJV|R^7?l~|N)Us8U$U#bmkyf7V>`1*<-
3D*NZwlKK$xp&lkO@E;hMIPP%^_;>hmTP}*olFmu*%>taz1V}!CK(?7kfECbuI}o{>D6ilYYdLUM8zF=o*o{_C9CWXmUX+CYEci7
ZeeDMvdY63(FgumpxW^L!v?)NN@yz`Te1p5371sc1)C^qM&{3&YTGe%7{#YPIcZ_9EUY8g>Acl*z>@hYQ}v$2<TU9cBAk8dgi%Fz
gT942!ZWhPhD8yV;XFL9|eEqV!mXm9nbU*?sRxqnglY(QubAre7{GaVXiJPlsda_>}_}`x+G`H@@%>t*Nz5bi~TF9-
n(*~@T*c5%A>MMf-
P!^IxLCbgtx^Z_Btp8{a1%WBjS{>&6mw;+4*>SY12~mUd7lds&sAlzlPvGfMqH%H=yNlji>jpe$mjr>KGA&f36M)(pQmr{W$4G{m
8S)V&D_3FaNYH%6*u5$}xn{mF&y-=U6E^e)JVB6b%VwhswT}b2gQTZJJeUu#rs$a)aE{*bE}E+^n;_pQr~6F7(`Ud-
gAf3K_=|ip^5naI&5Mi`zBjo3aRnWzZZt=Y8%IR~9T`K-;lDh<7!3M&kJY1|+)tC5vhAwRQASkeXiU6JjH030zf_#ig-Vb3A|OAK
gp%Lh~_JEq3~?AjCIA0c;8jKE~Fr8d4-@Kb%5!6JYL%)=@5dfVX{&A}h^dr+Po048s2Ty)Qo2e4b#ut8Ip0I(l{B5Ersl-G6J~-
72kGW&ek8bAj>Lc>Sb+kt^C%f13BU6ezlQO2QdoyY=y+rqgnSvL-Kje4nHtR`Fs66094X$)p&jmo3Tx1@zkslb!0D1GhV+|Do{rt
Q#XafZ*&_Z!G<v?4b?%8hG;j16I#2{kt~_Vlm@K_%iP|Pd2EpveLSIuKN}vO2nD%-
%EG;7a$624li<xo`*e~QnSn*GiQPlzm;6J0@EDy0q6d!K~REHH0|LFh7442P}d-h>fz1ANjV?<fc@sGnha&)DkMp^kjB6-
L@#EgJ1<?HlhcNFy+BoWx6<rW|Bk#RhNe>Dl5P68T_`(aR~NrKSQcQpi@;%NzC;f%UaA%*0EcHLnM#s+tBeRq#Jy>fY9lc%e6@{Q
c;%upg*mchqMq-ZhK4n#5BLGPp3PAWXR+h=d{i9Ge{wCVN3w1*O=RQ%WWw;un-*_qjO<;l6Fzdu*^;9hIDp^1-15-
Muvs3e6RAXr$^x;?rOuc9Sml3c^rpbKQ5q2TAOk4*Zyu8_t60VO%RIiPs2gBqu&My9NKTa=n&8?Jd)aV6nA(vjTVYA!(DZ7+9xWw
?DBk$@ZdNoE51p?`^Kh_jY@jtFAf$Dn!B&9fs^c{+Q3i7`>zIIA;3EuEE@;{7U9*Y0{S7k*TkIW)LDuvJjctw`zc4cGea+MM?W{E
=Q9dS&?lpP$e5r9}(8BQ**B3*TK~v!m<fVm-
r<flRZ%)G64G_yWp2XGeTmW*?F}tgR3>)jM0m%McHiDA#ZG8deLU3K37c64;M?OEfL!_9kHQ=Tw*SAZKBPXk_A;ei{cu6*}YqL5|
QIJf3Jj$<K7^k}BgT9UJCA49cBiOr;6-Wt9UXGugL3@H$x;WRs#@jiGzftZOfOJJib`sBwubH81Gu7q;2t7nMjF+S5W7-
qv;0iQ7@yI4tftwUUiTt@oEOEm~)@a`yMrfMha4FUO>TcNTzLSoNmreieTqeslfc0xRCZ{5~MmmOoQ<l!%$(Xqh;&aX%YZcq-
c6cxqu*Ul9%g&Lgs<|$q+W$@zgIdCD=H4+Bt#6xgF-t#<9llyB_p<-cN4`v_$s8aH(zSLT4YyZi-JKwifLIM(Q~Gh+CkQeHSfQ#q
yzXw;khlIevvPlpAZ`>|bS?edbjnfcdmS+JGEGIh4>*KivUv@grpwiG=ok$a<p(TvJyurf)yNHJ*lpJTptRIvg9Q)y^t@y!)UipJ
!$_L+YkC37ip~Pwr`?0M`8KrWDbqW!arig)FNy>U1M`3^N6~O#-U596!S^v0MogbxTfi0RCapi!dtfC*8~-
b?FM=Y20M3{;Bc`Hj=@{a}Tf>EGYsf4*)Jh`&IJ`Iw!okcrq7=9e21bkxV|u5*-
Ix78hyRgOW+LIWuSj;~tp?o=<T#8$c4;GikS6wda`gwbhP4<fqyAVbar9=YD45i`n?XUrH<kUSS*oi-
y~wkys+}VSurzI?X0;Nb5;20tRx<5(SYi%3#yKLsWPD(?<Y}Am?PPWcv2-
{>G>RR3aVG2rlRUl64d?Ec@2T(6&Z<6sVHw(&{Z!;8qtCdM58?rf7)$Vr48V8GD#)YYP8R%aPsLG>@Xt2ILHpE0Q>0#ta*a0)^LO
0ho^ONBD(6bL0v{4)H7(0l%@Rq$G(KderC&_K3?HP^b2O3Sfzm%2hNz(r6=11c&8}cSq|JVLWm|twmsThU<8@Ot?YWdbNDphKTRH
fgpUy^HReKGN8USOk2Q><DN8LjNE%tJqqCwON{2x@wAuH?2idl5#SQhykxsi$}mXI1t1udR0)I8Ew6X|sPDN(yVES|>G*bhetKu*
mJ);Ctee}B+)WNU9f;&Lv<$m@Ph0({q${RloF)mIsOCyP2x7@U*ad?Qd~P2-
jrI&U1AZ)ib%QJvAxlz&@4WGbLFkxbt`vN!?}>A#Ol%-
X(UK<PuvC*&$$kH%q2U%(0@{oaSBbA~JpX`nHobTK<yT4OI;gCO%v@N%KGGmeNuT-
@=!D@L|%ywb{9*27&>hBF^+JjwttFP**Zm4}O4ptnvR^^=9Jk*YJ{<r5(YI$c67+L8k@Z;Tla<^=A((MRW4)cZ`}9=!Zxu6Zf7Db
W@KwkFPG<0XqC#r^1VQcwRr9ok2%#wY?beB6ub)5Dd=&w@z3srcieZc2VgT*TmL08V3tRLd^pY~QM2@1;p8$b=Yh<LHj&%*u;yo=
grsL|Fr~#J1CNNZv-KH{^%&-{2n2?9J3mrPRMV{Zy}GKg(gBpeiar&OU3e<v=@S`SZ`q8MI&l;Zcc}cV6-gStkG-
=%DY`d9L(PX_wW8lqn_kBo-h0o&j!l*|-x>zE$asP{G73zZ>B8NP!W2cDaY-AW*^0&8Wd9*9{N#69Jewz>rhFzjp`DfK?&Qc_(_t
2x{mN%Il#l)2q<^l<8d6AT%!mXrzPrxaZ@pE~{9M0!xw!5>D%PeIadUhwzB@nc8m9(k;O{_HHv*!$7%B%&fnv$_X>y<t36yb~$Gg
u4Bu>U=%8v6<D5K@k;wK<J^CJ@pid<hZ_9K?&ONSw*N$96J7|L!Rb3O+V%`jOj*}|s+k4sP`p2P*!!$=-~0<u{`2DN-
u(SE4+!WE`J(>YcGn^16h#SKm=S^F^KVpmo+z?NQKN^n(XbzTIZ;Bh5L{@Rima4BE~65k{y)wfi!elqnqU64y+<y(=27Yo52hryx
QVcNb$HD%@^b`LGvp`o&FN#;fVJNurgwh)^7q`N!hrn=y|1=F(*jGHhzaxJ&)Nx4=Wb(MLTF%v7E4b7D<k!WKqnh(0G9~_2qf*%2
FV5~npS(px+Fol3#?nQcF~j{XPcw{$DmNZ)<Icf#26swKzm2iMMzowiwzA;fMfm~o_NaID2Lit6ZB2k;>!t%1@C4({}U<tyT--
ENi-phs~Z+a*>5xTJjw)#hn*~?nP-
p1mwUUSJr!p^fT6O|oJl@!cEK)r!K*8PI!e<SkbG9jg?y0Tew}{qYYt^RrIe2ZE@W84lD!0^TkqktgTM55nZG@`sVZx=aulhk-
3su8NY|W(!A3Fy=Q`+v@SU<Oah;QeHy%vlt9W9f%sRAvsoR3`dn<!FmfIYw2gL3Jq4&7-`{&@nFq0^{Q{(<-
bG2#|PHW<ODQOlFt<##E@l8MeK&oV*RixU`5aYP`%jMO!D;Evj(&zknUsgoUfMw%r3+Q|-
4`aC$OGAEV2dnZ#v7+@1vt(@0&@cgPhil`Uh_~p6!SO%cE)EMHgL~L|7AWIQ;9^*IGUe8d6B1pig9F;h(HY4h`DKFmYMsYks)cPN
c)0Ff`!ObPhbt3<x0HE1ib`5rqo65z{GG1I3C5OSIyQzdH7K%j$KmV=Q7wML8;-6qKEyCpsgB}-
|DX8keM@!|D)~_kfK3Vng>_Z_!{C{rSNMozZx~@Y`ND|QQFC|vjmT`-
5JLSLk&xoG>VN*Lh`n>rP)lVXJ?P=`42hqSJ!&seq$A+dy@_a8i+oVR=e+i+6ff#UJd;V>^{`O^uz7VtdXg&f>Ns$xOvVDvxRR<g
P0zS^zA2V`-%gh4FvmrTbl|vyNJoH7!PY7nmHS{Cp-WuXN&EpebVUG}&Fz@k!;@`X?51oL0B%8JE-
_)a+c%Db^vQxQ^X;R4{=5L{2cbl{-
ks^|Kehp*@gY=Wtm&!zotqdo&w9|$DB~^;#?DiKO8mEf>~0&i`pMHm`*0cAhVb9$z<mK!fDVA&B!|S^VE5$2tsUQqsL7@v7bK6yl
XrVHlqeJiF;L2a#F{2>bOFP!mzR9-VJjt9@~nE1gj3^WV5tBUC6h!2)`N-
^rKI&uyev3R?q(p7V25kV%&H%vRz7tJ*e3UU_@K*CpI&52kXT`@wDFJ(#ArPK*veFG{Sr10v{ZXNntnxp!6iy;+?G*a-
dl>sSwH82=g?I+cSi}vT)S8-zL)Oq{nu*KnFd)r>>?{3h8jx+M-
kJ3JEBIsre49p^Vi_B1jipCC&*G*?lOt!>MtH;EzddAJ(8~iy)X;K7PbhAz+evERy`Sd*frMYGCifrwAdq9DCIxQEMO>pdu<o)U9
Q06`M@hd-jsLKLG~ngbR4)%xu|v-
gxm4Io~%N)d28&?23#rFl!u^L9M#g;<rZ1{J|fNWsy!F_`aDtWuO<b%c^AkKf}n|rGR|R6gt8^uv1j-
!?+cDBx=5dfDJBPp|B?8(9bV&_DfG;HJYsU3n8`C13F$Elrp-
O*nYCqKKRj*|l|PprvqATgVc~mIibQL3E3%JKQ6LO*Gbn+0$5ZNHX~y37O8NOZt3<5lz`CLESv)n)anA44&+@zlsVn&HD<hA2A`b
5`>qmnAoS?W}lY4@c%*0}I(GW?Z7}ygwCh^XL$6fL1B`Z{IoV-p7N;M+=06HjNU+kT{CV~cdm=+`A-+lq-
(D1AeoazjxGzcaaMzdGMGV&zN6QY5Z&i3-
C!sHvcgByOY=%ka?EvFi@ld7dP*mqx4vQ;ZuX`xg!_J>&QMFVj*@PsR+A2IC3ilp1$K)f+<txC=`7LXVIy4aYL1n&hu-
a#eV!BETkY)ZT!uf3iru3uy<WEV>sq;>SKv`BFYwTz?Zj<3fwaYeKfCC?`S#?PgtXWy!&gRa&buopt~+4B|Y#wSf1j+b8uREb$+H
*{uHGw5zb++AS512*pEIVf5gLO)r&?rO7GxXQK(oJjvh+eohdk;-fH<x9u-
p@q^Vnjg1Znt$E_ZI>PPRb<?#0Y<9p1^wn%!}C3ob{AZ5TI_<P&nE#U3S?4eTd)1fd@UTBTBp|h1cuqJ(m-
@Vx`iLwW0WSbS^QZ95YiobO~jCe5UjqEu=9XA!Tifa3!OFEy2GDODqml<F@~wycR!{4H&EY+p@n7}63}qFMc=@rp5iGqlLl}ld{!
hDpA954`IdR1SdDCj(e9Xtv-9ok*be^|ze=`mATTA*#T|7y^`Cmmo{Hr`oG^5ooM-jx+0+<I?~9f8lkq2QKJ3ken$zg25H09Y(mk
pc5JELJO|JoW@%+5sELxM2WQ&RXn1hj-H$KuXtu;-
Q=^LZy+wu#)%vD&cqN<Bh#ePwGm<Ht6jS#fdj$xSb1C2TVC^>3Lbm~=OLh30S>Y~6IbGm&LnLaFiw?K2t#o*Qq%s~U=b|3nnaU|K
<`p=9xj`o9e;pqASzF=X}*Juhr%N-_I$^fcKm>D9gip{-
2ht9g=*~yYTSXQ^Am?g7H%zoD&xx`}dkaN5UA2>U(T^q`pm4R{)OlHeNO4!C)SqFnWN%8sN#R1?+)0FJp3R_3-
>e&l#tXSPZ{pvd9JTYx7AZAof%xX;m@%+-Amu-%-(`0ja-Sjp;4*;<SRnvO>XkcRXS2ix^{8|}Iy<Q|GyznVjT<D!61;woFx$r!8
k5(!4@PS<B2pLUM*_@DL6|-
T_joX)LI+CAO5;k++S_`w;^a62(+&dLhP<;~o3af(2*PUExt#qPQxEt#rZ!`?bA4?f%pJ5!!(woevXlb<-=3Q-
eCeYRT3Tg2F05hM(0Fr<M80n5K@w0lGwr>F?QQUJGr(qWM-`g|quP*n+QEdLSxZ1chZHrLF-
z+O!t<t@GQer?Jrcd*&8e;j}{4WjaM2@d6`Fj`YJ73o-KmP8=->rSyf2cgPhPHpWDWRSip$rmMA3xGUcK|diM)^FnK=)-
8UyPwyrySAqOEEA3xh>-0ct~tuPMmmUHK6c;oFTPZ<a%{}+#Sd9EG=)2JxrZn-
0cEDLYha=s(_2T1@_bI1bvCCdUINf+xjh#>Q`(|p!JU5wzroA5B&-QVp+b6&c!_`$>BEHqLDPs4T|jeF9kPn-
c`MOV?vu~2}$bw%j1w$s{C^ZePY=_Q5Q&K;^EbS;qKj5Ske^2)&{8tU6DcB$jJ5D;yM-
d4`EC2koQu+Vj3>0SVVFF$6_pa{gR1Z)Yll}zo%T}|AlG<-R)EFK!ngzBR;ZVKXx8d*h9lO-W}O8<jndlAVNw9gH-
T77XNt9<7qZ%WOcxX)fak$NPXlRF=x1hl)$0Va%|DvNSNfvxj1>e^EY27z(gi5C8^w3U=IB$OrEyjDN<lyFuxEo-
7)UH(ycgfZM!tnhVvS>(YS!OjFb_Zwm!tKXtKf?A35%{`Owbk1%XV&FyUZ1IB-
z$2pnzrv6!#{nBYDTnuTfqzysRf%2!RV{Oa+0I;FGxrsb9QSAvpbntMc!rDIf%aBZN;)7W~UWJ0GWbC}d;T&&WzmY9MJ=Px3ptJw
HU#UmT8^x^aOjjdvPt!SZ(43t$opkH3rM*UZVmwCaK2@|t$5m7mY%<46*uGJ8D)BUBePFeN2hR{>g-T!lEa2Wb<-
BD7xs=%f!;JCvVv_3o_zT|pyrCjQayh7laT+P|JC}Sc(;lgnkR_@$uu44@sx0Zbj77>Ec8mOUFD6Dh(gpO5>oIFX}T?Ou1R-
xE&O?Z;z7pu1)BZOA8kGx5STX^mIrJR?{IuCrt2A!qF(g;+Mg+L83+k+K_r7t3I7jCi3HA>e;`@)Wz`^Bj}g@6-
d3aN1UW#WY!217~m`mB`Wyn1i#Plg>0*nMv@nu;XMes)(i3y-H?1t{eFh659$GF?36mB?jik~5c>Y-
dW6xboPkKFJOGWg1F<6CmmX`Gl9}g(Bk;_uC_=yMnL<JR02U0~_ub`O@_c*wm+*4?PGSz?reC(N+E(jo^N$ptf!qAvS4}jBxMi#G
6*p4=v5NlMiQ&fAf1_Yb-@7%9i%#$|0KlvQ5m5_SN0eHD~uG{T`;cm+V*i9LUyaqv6uOTL5;f_7ik1>R(c|KVemB9>sD-LH%hAZL
`$CZ&X`A7R<ck4t<wDE&D+5tFtlA?ZAM4ABZD8@9Z3Dp;SxEAgl(Ks=^~VwBlZdv1Te<QUi|gp`d`cp=7$KeIn%2I=^m91WFyVqM
LW>SD}_Jg+jrW0t}(;K%$GZ$mUUHU3C!eI)btom5|`JH5S`5SPBR;khCNZe=Xtq5X(Hm&9uMa>v<d1GE(vKIzdjxTZ_jxN91;upE
fwhnVvq;(R&4o|DKlWjQ1x%i{I#H@>(I%Cm_?4rpyPw0B3f3X(4TU9cZvRhYdeY=O*O@_P*jVZ-PcUQali5tZEZ233pak{Vl1#SU
4vJX^sY^d~12JX)HmsKLUa%BZgo8RtWC@$PTSp0}HE>ib+k<r1Oo6KpM)s;Y>_Jy3_J+4X{N_BgLS-
O9bE`@zJyv7?$hm?n>op|D%XIMR<-
CZVqJ6ddj#OPt`1#n5oCHsLs2XD7}XArcb1{=JvN0NB<TE!x7cQy#!;;X}uR)9ZAmDRhM$+<~`#uc)K{^Io(@O?cK>P)tT~y3D9z
UekEh(a5%nfq!>}SdiCyG=yGCY&5LwlNSac2K64T>r30e8Kx0OaddjcgYJ2v$$K`2Xu{j0}#Z`f5-
+Vo6fh<D(jQ$x%FYNP<40DgGISPT?=i7u=KHuGM5VR<LF&gp<m~iEl{D0YUtPJ>l&K%J?#yd7i4&S7>{gUt)t|_qon}UMk*{?~AU
iIMtdigoM#YRvu7**x{2=4E=A|c>m%DPV-3d6-H#x9M~4&(1&E%}dn0es*6l!_hNft9`C;er`P+_x`l`WGDc=_gvb>h33n-
_+=mYb;H{00xe;iz@7l4)b>S7bE~<{&>CkufY5O9Mk>t2TYJk18whc9Wzykyz|AlTE^F1AsoP7$8wFUPT~`U0q1io1D~=xQhjb*s
kY1wO2F`s*ZVE5*tw)^7KZ%)G95dWCJyXxLa}pApc$ta>kt90iJH3bW2syXiHt^rP}5HK{|z-
0d`zPp?3(tRyCBwicZd^UQaGN!WiPoNRVaau3TT+e3w$!jR!p@F1Bj_JQvQj&LsDTe`A1?<W&}P2yLsm)8R?v4Oj$}~dxfF7q&HG
Ga7z1PgUuBML$78<1k<sR{*@5I(b(pgYBfk|yxs0X_|~J*E|S7Xc7D!*0s~(Qyl38#(uYg*{S?0fd%|BU0oWi%dD%WMltnV4$L#f
8=DkG<jCh>Jq1)yd6gxDS5v}0&!ZiIMEHRXlcTx^8no&Si8UMi^UH}v8a{hWudH$rDA=X;LEpWJJGbZCDn2AeQdc?e&>xT;HU3o2
^<w|^r?txUYWM<n-TS_HR!)4C@t;YTS!Z>nu!+>VWqNcpvn6yv8n`T8<9lCHA(XSUKt2|(=lgYgHU-
B+p<Do)gkMSCx4<Vk1vfy5l&JD>HbcN+S5RtFOUwqWRD#dz~D!-
hL#pWUF6LM)K1nKqY0%j(zbhmeD<n<N;>n9Yj@xMZ(T>8$p)wdFb?{a~V+4?@YCr<Kg0qbXgmRnDKhL*BQtPl`tk~Y1*Z4ASWY^u
|X>LJ~eu~q&0mm5WO<AJ`q&#sdkW-
Z5GbHAQQ(_Z61sVQ()V$OInN&@bD7^c*bpdhdi21Mt;8qHa8ibSR`k(WEj#;C3)wb{l9ZJo{DMnCkxJanLzMR-
!Qeo#@`gVd<rMxl!&L!5h|xqxTO=csq**vda=iZo43MB$8Y^!G}GB#4?cCiKqsaF>-
9q8rDFKzMvtt^ighnohX;H?J@ot|wEzYbmigaULuey7Kyv`@TH8i$GKIm%3K@Z)hihyAciv7x~_vq#ehYPiY+d4mj+6dEZr*hTvX
(d#m+kF|cf<u+sPSws#Q=qla{JWydwfrtFPLN<Bp^{H8M;$+dA>b3D^1I0JCMAK9Az3gXmQfz9VGX%Nsa5vJk|4K{)X>DaU9^Vyx
QS*B>eO{~Jpjt~6m)H=5tj2Cj|ni514@xHKFS2JQw)Is@d8+>ng)<VN@D3r5ngFwe&W*0;LAC+`1?$&*8JQD}zubfIiqb~aI4jSc
Mf0+qZ-4|&SA>q`boqX}NUo&a-
>s}5SR;mgCP`+lcQoU559rpge6s@mM<=3KAhJGT1EcNKUACyLUlY{Zk$1Q!GxuzvFZsu?fu>2qM1;*j;PCaK%R~(h@Fpd<v*Rb9s
Z11C%m|Y!L9Id2*F*oRqeh`P3>`~h^Hug=<#(XXWuq|u$;4SNiJ#B1xsz71!$sX&bg&<B0WrwU|%P8l}tI&tqnELz5IO0cdUQ)2F
Sc)+ZNZmeZ$g>v?q!~Vvlm~Z3IF^_ZEQ}sJd%F?R45wQit5>08fCt;JHS#y{bvhe0)-
SbmqLZ358j32diYT~0baA9F)~lc_Mu&d+dVHrnh`5xbN>hyhOY<DK{jQ3;`|R_ff+unEikVr4Y+LY}3dubk_uw3#vy$k7=krf4R#
I4fi1Hc|5=0)0Oc6;O?=YJwn)&CI(}eirNZFKDE)F75gDn=_TR%Fy4G*LsmRgNNT}#>f&r8}?jMqhIEsD<6bCIw|DPCZX`t?Bp(m
YMI(PL-evF6y?kKHU8$Ku+#A>ga%ufq_2yNu_7<@y!GKlx(({d(WRV<GJ~^A_a-PaLjN?dN)W7X-{ofgP>AObD;>IQ@DJ5-
`$kRpjr>;Ji<gGEdg2uNEaJHN)m8G+D);xdQ4=(|>>*eZH#AH4NBdiz=ST*iRwhsHo2|rIpxX6q?bgUyjyT`_~p8tccUY)V6KE<h
(N(&PeIiDJ|LPi)+hBTu@1*C?%J9!~-&Mu;$%k6<Oqj3a|fR&{HLQ0Y}g|NxH#*;C-
U;{w)mgHABo`vR<7anE_<6<VMMDp&@94Y1T=_yfYt{$_5k3qz6I5q!~8#C>P<hbvwb3rorTH<7Sl3yDT{TZBANL_?S73h%L}Z8P|
Q)&Z#h6lr59^j<JN5PFV_!{ywiYH_9Pzg91dyE3!97T`>P(Yocj1jeuLw)Dl{s+Yc;SvJnYS34cymaaQV~bAfo0AAsX4wtwR!vA=
2SYe3TlOuL9r%X>;_=liPR$~wHKUpwwHN+iU?6m%KO#vlr5MR!xoD7Y#LayqFK;M5-
gtr~B2F!J$w;L=O1I#&(LF)C?ZvP~+Fr$l2#AayKRQ|1+gn}z$;g9rqHzvfeo!{7cTasVB=R7fb8sJt<D?ef9Tj99`mZG{9SK@}f
Ue-60B&YrV(Tp0rp$tlf<Wzkz5V-
nUT2!*=YOjVGrogZNHqISc@%~{R)K*RfgC7oX)2)Vd9X_gy#u8v`G4jru8bzfe1Gni8y;1YA-ldsjk@UL}L2@%L!4Ak&*D$Zk?4U
q@eXC{%*a4}C;i9HEC0%vc_X)ZUi`)1~{pp|IfJ!uu@!*3Y=74T>Tp@0c(7K+D&z)?av!zR^84|Yl<14*k5e}VFK-
`w?H?g1W0Gp#mWY~uwR`43uhn2AU;$qbUqN1LMJJ>@&2{vBxJMWpx~JcL9lmBX|?k$SOX%=nm0-
d<9&G07l=z6QKkd^9|@4LK0N^3%FD2k#2dIIyKTsO_@lMCZ6qzGIhmGgI1sR>EA=7OL)~t$iZ9_dHYd=1UO49@cLG)n>&lvuHNBs
-J*l=a#OF(hT6HN~9nPi7iV8G7iq?h`w{{PYg`9+~@(z+|opf2V~FT@-
qnw@NlwBHH##?coks@Nv8X%w2m_{c`7=;QP=>GkypxXDZ`g!7o~Dm3POHGmXoH4rF7dv<l8KWNr?#;dYSdK!^H*=mHL3qzqlf6F}
fG2HiKm4Y$s2(j>LUmxC_tAdFUbPiM;RKg<8JTk*b%Lv!6=<yI4A_YgIA6=)2@6l%JfGCvnyo8grr}l^I-
ZAbEeX#?Ri9NJHdGh0;?`U+4T1JHP2p;ysB%Ao-
)uIeqDw93qNTEi~^@rDJs3N4^^Jr~VdiP(MB*<GS>8&$svHv4$hTu{rHyfV?C?&h$=6Vv;W=CCr`UW8W)^tUueAE=vAO=RvX1{i*
}?A6IuU_OaP^^B{ljqcpq5d~TAy(XyJGovk7Q|5%OwyZtPT8G1G`w`h%-
BTB1b%zX3c{sm1hKG>Pb0O#ED2^#!*!?e4pA!RU*MuJ$wn%O(S^Wp}~p*!Ai{p7F*8>D?w*{we*oOLdj+Xe{~M@UO1+yDN}0OFq3
6nm5FP|f^9(_TO^H*l@X?t^j<gutYhVM@o*`;S(xaSCiYHvg`I&q%2?;Qm}J+0)O_W(%RkjFQfGhLinQq6S}`DMu)WY`06iutQBH
^mBfcN1Jm$Q-
`g@hl`~;nr=c1U7fioL&Ec<J@y2+sIiYQF6cSG0zhd2C>iP}LWuOJ5w15Wfsn<3*qu;p?>Q|0{&sXi!tdJtTRg6+0#-
QpFZkVLQGGx8Y!stw!TKYs5_CV0VN9XaH<Z{w<{PtD5*hyxyq32mnh~V~WN2mGBW@{_omM;-
fZlHUv+;}Z2i016mslou?C><*K5xI6e|YZL4lU3)hhZi37TWcK2h-zV-z9jcGyr@~mI^&)kzD9--
X+=0nOx(F(T<4<{ifcOg7s2Rar%Eyc|63}>j!$r9V3u1i<&Uda_-;NUUaP6JnV3+pAPOe?vmJD1z^TCM-SH`3pJiK!F1_aS_bAbZ
OAq`#*_m@omtk2R9-WoS!D{3Sx>>yJL0{htT*35*ZX3z%kk5eJ>-
>F$F}DG95rmeg1v2k<MdSUxM!$Rr2D3~wGY=$=&wM>vaE$?bRAiON6)>C{0I)olp=nxBGWL8c7zUWOC^TMBRD4xeO?HNy5q`}d$)
Hp{7GD2Vw};VhRYa3+|PeQo=O#%Iqwp4jY)$|e3*bMS#q`x`&V5)>dyewO%x~z`o+)!n{wopGm}4LJFTkx%>T&U@1FN=!bp<I6eo
g+2Y`l|2u>^K&FVPePA*>kmG0B5_4H*L&@t(01ADE+atS7Ga)^qp-
Q4?a=}Fcc5OWZ`>^B6q*V<~$Vx`r{=KxwrH`4FRkR>odnQ?KRgA_@fb<VM3ZJ##?(crS<e71%5p%H;dp_q#YugLH08{&>@c&ZSpp
+ZR{q2~98C`&q%8u<)1`k~ppH=8vU4#be5WxFT9S<BNpwo^Vm@$9RoQ8FLz&89r?JN?3a54wxJ;P|+84Zt=p77@8qtdfk(=){^Ic
Y9#T0#j^#;WC{*86BYtZ5|iePZ}GZwoMPf1x|$JhPBpgKC#Dnp9&?p%L&fED$*;fbqEn?pZFq|nX4?qlGqysH5S2z0Whr9TXs4^c
iyx&CrcS|pY~ZbI#-zYH*JLNQQX8K@RT;{JP<l+gA&z(&2MO_(SJCLD=DxD+I-m~oJR0SEn+?;A=%>eJ)lAT05#qp1-
`Yefkw&w`b<xTE3gQP4GvARmdK7l#$^SuM@wXh?l$&7iJ(y!td=mYySi9F(%yf_GrM1f_RtuoLa;BT|2s^T@JzaHu0{(3f<wL6hO
_)%D8i_X0Qobf-74(kA+cOrkY=9N!o)zj$0U!jTYmOP-
&Ti&D1^uuD{ZHxUXbt6xt6jiDv!jPQ!sAi^lR(O6~KsbMzxxOg4Kpcu(Bt6a$b5mwWs3eQ-
A!P{UYR7?!(^|i0e>uSq)mdTCi@WADP9q>dUa|6M{F8W<BXLB}DO~Lvt9~R{H?A<{&K#*zu>?3$2BT8msc8+=D3%v>G$5;aM<Ok#
4fInOBKP&LXRTAV`=b!lW}iq46bheMXD_T{{aD6aAH_`wAY+$utjk(j-
gvxHCZDIaYGbAAv<WV}}LMccS>bew;}{^6mIuHC=`XFgSY(M@MiL*`fskKTp$j<qvPc0S!K=Sz<t`RUIxBfX;`UVWktD#UJ3@(kH
~rHd<SKInU<zz?bHlq;OEISF<l;{8{LWY!nya*=f>DI2kg!#7xXgur(pQ?qcqpICI=6v`lrq?D6bkw3n4fHkqd%N@g(q3w^6>tIZ
vL<;<nW1m^3!-S204l3+vS<U90p4AI#tSUko}dh-
4dyG<tJP;hLR!RN>9a^w(M9I+B)eZ`Dao1a1EdOe1g1nw7mDx2z>)T(Z;cO3UECW|W;>?H`+IR)PDeUNLhP$MS^xZuociKx~-
87b@dcL3_eo767%HT$M_deEBe&<X6~)lA$~|9!28VoNx8vZaZx54C~7k;Sqqj}f7~;~9Rl%#h22_iAND=jvZi&Gv1CL}x>nuzDe1
RdJEp*%v!BZxXltn;aa1#Suh1(<5D}eO5#GjWaeK;SsRn?bB*3vL~wSjk}z>YJGrArS3<%RKV>}&-
6+K^<*(DX{B+dYYjrF&>(;67myl9pDQh{uU=-
&mme6*nZDa9Jm$+?=jeDjy!yz$VYxsfwjdWVWSf?}`Z^@J<pRIj|9>&tUd%WB;s+qI5P?MuxmPH~@E_QIh-
M#cr|C@`?AR64?Hrcq-@`|k&#-?Z9uNayz-tQxyBE}9Z4R1dSfn<17VW!^Ur10^*;tETAHJ`OQNb`vkPIY6ZNZbSp4OlXWr=v+d)
LFLqzYoNa3%JtI$a5oyDPid&H+pEgtbHy40s}F=7ZQ7RBC!ak(T;oe=g#5S@LHf>OBlY(~dYbB7?7Ts~h^>Ahbb+DOqO_U*7JLXz
@QKd8K=Xq+S7XtMJU@_fs;Xb_#+V0Ywus#yx!$uHxYE<z_Z{yIz$UR`WiY;^!;rFIlvIyvq^o6ul@ZH@6E=7h&)Tq&T7c^5i{1Jf
~4v43`iRuiyP@9FBwaEnei`@jj<?s+Z*;UBWco=kCb;37U@mwr|9KnoH>;ovhZ6OgLA+a~lD!OKn9A;V+>*kmp0fkh`}oWLs?#W=
>%KoicXP=7$t2U2=;GKrYi|kL?Te_oGHfy^+3+{i?<7o$LAMlES7668ZZq$S|}yO#<b&Robgk+vKY~kUOW-
f|SRiW}C$ZZ0uZkX??z1*<#uJq!1rS^Sx-
@O><@`DXID14FSG_slSyE37LB>g?*xX7hacyG*KqqwhJ2PM2Y$A+s~MYhb~Iu7>QY8GEC5R;8`z4lDrxq+-9+RK5@E1ILu-
{de7N8K_53o_@To7;j0X%B!)Xh|Et|QB4)E1Ee*h6qVtTbSZ4k$hQ43t0hB*2{A()jD3zomrM@OZFQj>TWH)utcYLC;rxBM2ohP(
6`^!b$4Tp{q{R`UoncW*9EAC}^Nqvuh%#b~A5n+R$F`&|MBLmc(*Jv&=l>;g4{Wyux)fw8Sn`5Tn4ZhC3siK7mE2Qq&+K)+}M&de
ye@JrOg?FIa)el*ik{!X<oJzyWlnUfvsivu}2=3R!oiK@VVZE3vyFVvj;8GfqM*7dRRT5Ux7#t2DIPboX<Em8p3bf~A1w(JL3n5F
t3)W1oh|XfOB6n4~55fAr-6R$i?(P8-p2OJA-
9LQVDE0m;@ifP<yIJo1&=o37xS7HA2{CmIw`&mag;Xl}>2zSUD=Yo1eea?(LzXL}D_&U;`8VO2FJihFgx5_V9ovzVA<9GJS0f=R3
$CY<m{3}rckH2nqZkLpWO1?}X=EvYsf+#K{#0P>L<H7E>iJ`p!<g$gSk%%u{(q#s#vy}%fKO%~-
(#KW0qtybIvW2_S&B`7kq|{|jHe!hQ5kijd!@8kN(nQ?q~)`Y;5#(xW&_f3b6mhuhpgnXtxDCg(|(unQ2zPC&mx#mReVHTI($T6f
yERnuk!ZCNlxtf&PRp?ubT7rr#gSuPqJf4Kw<{c(Z+@kPaGzZ{K($XiFB9Hpa;kikq5UlSZ$U_<vf2yWXj^In33_e=+RXtM45-
8&TsqRT{Iq(Eu+$9S7Bc!Rsnu_Vw=JubvgyV+1K=cA)SK+WqXkfwC@J;S@1QlDp6DZBpnuV#1kKn66m)>bF0j@DI@S}I+|YwWc3n
1T5LjE4|>Q#8Lj-FaykX}`3URi`777ao*L@7G_^h*smvan9Ra^*h5C#iq2S;9cb&;&x{A84hJVNAIe&*V8htRMj%a)3!=-
5`gc(;*#B@Q(Ko%h#lZF0Y>Xg7us1T$uO1Zly9Q8Vl`ArV4XfdgN=%oZ)J%@VIt;B3GEUm6Z@0%`nc0&PvM}TyGvwh}y+PtnfQ@p
;(@%8Gk7{ds>I`5TYK7KOu0L_V3*bJ)B1PxNseZT9F-
L>x>(Od52muLDEGe9~87Wu&}#=fQevOl#7V*Xygubz4Bnoeth)tf4qJtvL`#}~hW{iq)X1bGhh6^JHUP7!>j;2gueR%w~Dl;M0)y
8Ix^SE1t5_A8E(ytE_<Yw`M+6rD9eHN|}j^jbDAr<-H4INJP(-
Ew{8Tx^Ss1!!~k;{jvpp1MY(+a1&=bOhI@g77o+7f69lJZ@Fh<<WE7U@{SEV^+Wwb}_}g<{9xN&KF^c8khJOzRm(ugX<J?2<g0ZS
ybsJP)@G&FV_vw(F|F{keC1P{1a2WR^L`d%BwO3VP&j3G|}wnuXnRdaZBr3ygYRqNdm)`lb7W(lR77(;zn9s$ZTNI^3L1Y>mKwk+
6h{);4RY)&UlN$6typek?~POS=U!LyJ2u`y_u+XiOE%S7F*Tb>(L|z8Y6He>ltz;PO}0{VXxHcO9M=kePx3BcL>svDv6KdOSv;ap
s$FpR!nhAdo1Z<MSRZ!F(03q>m`y%Gx{&v5`#Q0xZrFdeZiu}wQ=ANJ?O(r9=~=aNFN+&1hG4~#`z|`+Z%Dwf<E1C$Jpo+#$v4H7
5@mTheCzyIOFPyc28BW`NK1?mRJ4NlHL3Rl;djp9VkCqFGp;sJ6?fudX%!5?^SePuwYMIl}C+rT3uh+=j{iVa(0$cNhta3`A<8c1
jjT;2(Ltl6v)pIWsXAKfC6)-
>=VyNgi9L%vL>UuP5dr<2RnD5u?FlkN<P#qg~Lc`iw{BQ#UhCUweXi#1j}Cj&~ezf88{YO?`qwFMOJF@3Zd=;kyB1<Ba()Vi|Qnm
v=yw&Tw4<dPBK_j#D+RnGkKVgz9)`vswI`LVUcQLiJ|y0sbOZtDyYYo9R41{ailDV0ubkh{(YL<XX0uE9!8l|{ALpymL%0%96Qrr
If*vk0(UxwryX1sf+hqo1MrczB|{!n9_VNZ9g=pG1xiBgc<pC@7zA}|48p+p=q9feqz-
&10352#CL;Kiynx=GlK*?9rIt+xXEmMVtA?=}6dL3^^rlOr6e%kEDwl#f7bR;+bJu^8PxAC}y*>s@jGu@PE_kbkoMR2oqF1ePi1x
Bvv<@lmD#^p2A-+$M5S>veXXJ!C-
pAu9C*NS)q+~a$77xC6&SqNoMw$6`{5@JwHHY*}qkP&%bm?!9SWft!hO)UNbXlxg|7J??x#D=4TXB>P1D$q34q0du;YS-hURQz78
YUn}gG)_`YzxeAA+k<?Oh~;H0)Vz>y9Hle#%y+*b`SXSiOgtaG~cq@4-
~GQ$AjfAGIw{2<@qK=|ImNyT4LXxp2(@})yTu=EQap~jMXOyV64he=d$SJRC_a_kP$m;xw-
@WNl!OWV!LdwMKD9`&@^1y7Rq{G4;<Z>4l6Q>uoqhZ#Qm9hbt%_a=(|Hrl)Okg&3O9^+n*{Z<NMa0>wVw=cU{rF%a<&^8#njDn6M
&<mAt~m2`t<6gv$0e%+9@`!2&UamIz+rh85cM7E~OihCN)Uw>jE28aHau11g#lk17gU6-
l*W2p!lbCJk&M*$^rb{UUG*10{<yV@l<VEPFXJb~i%hm}7bU8>moTFq3t7&gWCFP*>4=&#aX)t%?^ii~EbE9JkpA<4KTK4q48Vkw
iC!V<?9f5;W^jZI&?RjK>BeqT$|Oy8H83Hdh0qiB5|43Ec)x3zU$<EN<#+6Nf3BMkjqkDY7hivU*rjjJ1z>KPlT3zi@QE>gseCTy
)6UhFct39H2I@^d>s23AzYE&plgg=X4)lupXDjYI8L&VU@%7#oZJ07TiFi^5A<cG4P1XIhwj1v=L}?->v?Y8>twB3HU;-
EqC$&+>qKz<RgVu;=cI>kI!fiK0sF_1u?H9qlYpLbNHu#UizIj#LGw~X##9C%}`vWty@o}O!BUFR+%~T7>+Urpdkc0bY$;f@9QCI
67AlIlorVO!LKw;$f%UoSe8H=_?HF(eG*2aVz8(}>wt^w;T66Lk9`PI)mBgOXO-oxYn{wgE&Lqpn~~MdFZHF?l7Nq~yEn|)fP}mW
cMYwT+RBfiqCP}E1LnQQ67P>Hsl+kF=|>nA*VGD_=P2s=>VCiwAIm4(`V|q2Y{+#UCn`h$>wh?KQB1J9ZUzNOqvF!W*UoP$!!;m{
o>G3j=wv@3Uflq&axKp1m#@&<T0Q7so<ZRYF&?qfXV<<5w?Qi?rNR4YjsQ4NiBk#YDyVN9;~pO(6z-0=%@a_-CU}DVhHV=Hc}t|=
A%kOb8<m>5x2Y+d8DYzDwpXH<M|S-Y^Ta~ng~u%Gfj-
t2`#mMio)Q9`g&$TsePtwL!D$5N*)>$sH7?C>ucdKO1uU11zrXLPT`Uf@J;t2QaJNN-@ARY8-V=Z~yQzJdb6p{-
>U!B}Anyr*1$ba-PrY`bNzR1(Vp@{5DP)sLjfX@D$Os2&vsI%I-
L490RPzvX(9V8)1T0RoCJ$4Qe0A?yl`ByfB&>MZDkjloGVZ^2Jc9r(p;!>;lWykWM+t@AmUoO^s!%8#F_35(1-
NAFCo16hZ9i|vqpJLxMq5ZTY8cp7lDv}0lov0uoWE&)P^CzCHFTv}E^#Prn3uhX=UYJ8Fg-mIykSeZG=rrx3&E*sI$Vb=%&arb`l
iHlwW0`MDOW>E@n7<P<0M71#u(D&R1(Q<?ooFRPT>77pan*i3Xjsr%&;UOvuq?>FqBs#B)z;X4&l&U%T(YXf6p)O0kLLDXD__OGX
UYhMb=vYQ=D{#snnMv|L}E4eUx5}jJ|T%Mao-cERzKBR5KGlAqk{iC>ci8l%fj<^^X_oSVC6Lfj91Uf0hiWv1jspmQ;XYd<Immh~
f&U;MkT;SbV&_A=-
BdQj~Yin;fjC2>mmFkDhcihZm#4!H(5n#$qTJ$)9NU70)_#td=j6N;0*Oh+@^zqaarIx!2DbQ5R~%Qx{|mhQ}%@5w%z&D{g1-
grT{(naZAnXkKFr5Eqa9ZLLwWEwiC3e-wS-h-e+SGOs&H70N4+o%DZE&9ZI<vumUCBZ3-
wd=o|=h^?T{_YoQtpH1+C%anl3C#<p79y!|Ca`06PPT-7G**Y+E3m0ZNKU!(2K4$D)X9OwLX2={LdCY{-
AK0#xNn}}QR(H?8{Qg8&Y&l}91~yJMqHBVm^}3c)Px3vx0<i<$B|77zpD9vPbsT(0lvCtULl+T>cXqY!Vd<l-
0<`$i1W#I9Eny0m3`qtlIMwt`xz3%UtVcbGB9q>OqtD$~e<r%$TVUMz|MoU9n{wNrBk9fzku6@CIGUq>-XOwn?Y-
@QWAz)LQQu`r?Mq8@O+OquQoyu{kQEL<k4TOJXP!Y2%tte)@nn{X?rGO!G(sO3xF`#LVy_AImxvbEMRNc(g(Bhs0`b2M)sMlCtR!
|!&Lt<(>baK^oo6Ta+}r8f_8UysM1gP7qD7>(%;_eraGWJ`gwwQq@<6aoA1cz`Qg|$jMW#TCi!Q%6PcKS~n~p(yawlQ1guR^|-
VdVo#l$wN#=^!+&1xVvH|v`AUB>xmN4%J;^KYY)IIl5x;cXZ%C$Jo96HKuh=FCKxQUhJ=a#U9z;k0<4Mc24s&!x#?;fkRrM1Gi1r
5~JnPb9*PP?}0o2bLmz>9H9I`2^dkLxw=uM6YKMN%PXeQfK3AM-
9azk!PT!s~Vb8nws9JmJ5(V*p?v^w`NL&+SUPkTS9Rzk^Ase@2r1(KVW#dTP|Dm!lKWBzV~G|zziFWu@`o4?)#<tD>2)P7vi`wqs
lMYFHNLui1qiCbV8y8)Xem8P{AMxc&=Wh>M(L!J&98$yP{PzpV9P6uZYN@Cx8|qIo#9=f=*<2q+e(SYO+}S$nmQn74n8`y-
|H?=9}@cFw+-+;_V`Qifm_DIf$}GWa;#Tzn(v?<hzP*J-
*&9L;QjgN(1TIWmuUiN5Sj0jJ#(_vOtXlB}5dVuu;jR3A38eRi1RTGj|uruq^`{+Q1>p33Rmem3cbus+3_w#fOA&h#8OgJEM`QV_
26zziNGi1Kk9}>$(=spH!S~xd1bF0#3@|9xL!q(2zhe;LF$(fOFPp?>{)<F?9_@)lZ2ZeF(0S_Ic*6H<wNOOjk4OJu&f(K<C&1a3
f7$7}B|I%3Kj(OlBnm9dyvSwkfCt|0v0N)rSY1<hsf7H2+U%n*`fY`LS_^0*fKj9-
W5rx)k$H{S&u6Qiq?dfS`S8J3F+8ww#nx>E<+ofQ*pOp!X0lX8y;wMw4A@(EG|&ztJ*h*y<kGCxHJlBJ(#sn+G9jCksNatxASPBV
fci>EU>GzCV;QUOG~BP+33m;}rSzyj<J&QU=v+C0l1Ww-QJBHzm>gwiC_p+0D;IX^84fXzZ37mtgwvj-
(fW(VyOKxHM@h=*DH9LGm)6GK!sm$LkGAe68;9^Z*>kX1PXU4%Qq&4{G&DNai^<FayM%+j;0vn(qI*&BrWdbWp@g4$+Sp#jl8_8W
j6=-
;&<%z(zO4vS$(=BHoNsscW=ZAIQz|(5_bir07vULKVXUkZ&I724m;J*+w)(aZdQ&y+Sug$$xypP9|UDA)(t}K0c0gzS>Osn~@rmG
4Md`?M-a#Ky9>hBZUqvDWKj8aI!&u1;6mEV#2Tk`N?NtKwh%#Pa$SN931eHBs>|m%kInzwFxYizz>=1dRzVLnd?ekl>s%)-
s5MG94w%MClN={SG{V$-51Yf?*>)j0T0;y&%-nHW0Wo@z2IqV6uzy*eIJQt<;XJG3n5ZLivBm)AbY-Kd3vO)fuOgx5v1{fM|uU<{
_c^$+rjX7h_KAK$F!DtHm<93QEWbq86WCD909M?w^4V1%w5SC?h>4&MFSV?q5J|ayHZO`*QgtR%MQjoY_zvdKh|jhm19CGspZh{6
ki3f5yIT^+`C~S)>ENMzGiKme6zo$^B?xi3|%B{bV1yoA5#~WAn^i`Hu7V2G8HutJ8p~5O7w%IM)jf(D(0SGGW%|1r&!Wl6lAzA`
&E{-?#)4^jAW%C+-(I_)4_Gdatsy6vCsj7rlr3GT+s8D@QnKz0NSBq6pK^s2d&-
}p*H32jOCbVsjor9bc)y|`AXiHZM!rgER;7+sU0v?WtHH-
6_Uf`O0aO+JoggiqM*p3C@%@W@0QG{npSMcq<$tAfG3Gi&$wlHTu>pdvfcbF(F5$;LT6yJ$Kj5p0mStHlh$YWZ*nMl&M;T$_F@bd
)Mav8!_e82+ILis>KRWR>W0IE0%;_KQ)93x^<=<S7kLK!rLBsJdI<xq9%+oFH46Z_DtmRw1eYD~)-
}V|6xX(U8k4R+n)J&xi(6J4sXB0b{vPfH`>OFp-nXQ^@w**Xxb$hkR&|sgJsRO}X0i7+{q=ee1_2EVlAx*2rc#%&j>wI=rLYN<Qv
%#&79(2q93ZMemcArEM?yLu-
6#l=5)ik9VticJ?sV^^%LcqbX*7oxO54oGv^B7CNC&32UgYV$OI;Xo`=?<wj3i4!IpK}G*V7t9LkG*1LlRZb&G!20kP$)2DZZ3wJ
k;1`kt|-ILYWucR@nYAXv+oiF+fOa)<v088;Bzg>nbU{IaUM}FrD@Lv-f-U-Xgzch1dNU@EblbiX>70%VwsTo_kFmUWRP|aG$tHN
eEqEZhJCztu@c+kkJKE16NKBcYl4WNl!Fu5;OC%u>irP5|;noKI=6s4&Y*H<PU13I*=a$m6s+aVh<gQE0e@cc@u4Q@{|peld3$~J
NB7Ht5@p|Z<JmC6`2Fe0_`v$-
!9j|Xl+leZ>D)Swe40d!O%9UGu<lucGa;Ob5W}ITVsO5M*V&`wKLviL|@~K|Ms;;WBRctRlTwqxV<XAAX6HqiX7V806~7xE(=Tk(
Q<ukOFRyLc6ez5MV5Uy{=?|j<7dk0e;KmP4a1QlEqmUc8o}e2pWdm$qNmH(0xJ~QpyfHtZ|KK|PBWeK={MmM{5GBda{E{aWHaZ#V
|eK!9-
}EShSqZYJai11i9662sAc08hj|6wf9ObNI}m)H7>(X6ZY^b%pjun^w;FLhCx?W2%#!49fr*+>N?Rr1wM6hJya32MLETJq<V*t*P6
D{2Sj|o4(}6S$rLA&l#Q!C&<-
oKS?XJLK_jD4Ur{v!cjmp`xT2@bWQ&9%n^f*>Ir2vH|Z1(Qmbhg94qK(nxfNI5mXKw2_dR|<AFeXaj=B($LvC|>pqjste`tf8UXB
F*tt{&x);eJT}>uid2Ho{BK@U3S4oLoty?KaAcyeb&|=m(TsBG(z789&z=qCY;I;gA|`r(4!D=gVDHs)qmChk#cd49rRF>$P|EYE
K1K91J4wJ^%Qrp>Tm5cmhY&6fyEH)iT#RsQCw&kv}m;O;aFC>43r?x(YJcMoj<0k1f1LA@CF6DTJ8u!D>rbm7%8|bfZ`yKd!lb;P
_!==<(?LD6O{^{VZ7*Y+2V8!$;ja964zj_Q=!IiC(wUx@$AwhaXQnhrucNiYW6(`S|kjcRTM`p6z?tOHY;uuw`?-
{K{q1HLl0cdJ*^_m^zTs0d5ht2XHv#^DQx$<L(R;UHWCMmHV_U*dC9zg)~WNVSFTtFNG)xe0xXh9u%FUug{seI2tq{Y;V($(uvwd
t<vzXA~y>I!v4^CG=>Imq#KrP=BKxBmCI(eRHb!6d*@851+AwN9{W6Gd$%xY#)`}UKoPv1U1GjfZnB{K0z+F(HQISAk8E?7W}Uc-
@+}#rWzd&KWuBY)g_&w`p^M~R^Fl>~O4T^>>_G}oR{l-
HnKRz!Xu5p!v1|x%HqoK>rXw1b;DSuz)jhIw`J=bi0qT8MK0JzKT7JR^$C)B?Hp<1$%=lXuH;OJQZ1A3hDvg+Cixam^_b!{LZl=H
edNqCMDsJ=lD>R=I$@iO03*r>dc`lp!yOwrW60kj?45R{)tq1=6!U}<egLo?DER#S#b03ymY?3`V20r`hDfU=Ni^Y~fWUnY7K4^c
Tk(*|4m*%WrRpQ4N!@E$y`{CS@%T`Jn&b^%#J^IWwOXC?L(3u+(CE4L_<QPd4{nJ0iXqcqkH;4s@&FY|*reGlj_JTFyah;jYBkTr
d`A-
qy0&8r8eS?=Pb6Nd**A9ml{TI4mB{$*W%LkhHJN|IP)qR@P2)p4#gX!Y|VQ}@xMY?Q2&p7g7VK?3$WXy}HBPUqSI)T0rmeCIAa8-
{$RsubedoZRR!D`k47UCdg6F9Z_5vu2`z>EY4*n6vE{@<+^{iA<RWXblEe;0aBn!QH%m9;#ZCYK7#AVaodg{Qe=dQMZK%k{F#>ln
8oL~@g4)a=l|HbCt=R$xIU1!z*^L|W#H;C^|oO&>7~JCjiq7sn7oC)#ohf7&f05Xx?nvEj{O+3nOpIjVs-CzcipGfZB>n070_+VU
*d#pGXQPZR;xA31A6U;MCuV_-4c5_D$C9d?r};%(2nCiFu)*bjiK2IcHIu^C<uVKmONrR_zjR<va4BvUl}WcA@$x}`4-{-
<GZKJDP<+&7{L^m4}^JJzzr7lrYfwML5s@3EeQ0+>XFo(tQ_H~BUY*4S~mf0RfZtpm%)QdqlJ2B4URs$H<A<#JZ<hZ3a?f{G+OB*
IBj06a-@bIZi%+2$EDFDoPF&s3U8b`3vefkE}{w!>X>iah}k)o94|c5P3xxh`@VSUE^xy$YWfOqo-
ETC7@DVr7Casxh?wApxNd>Ni#qdtNV!*cDON{v%+=v$?%d<*@|O<RHp#s`OD7v5IsDVoe-
xqp&MNACq?+rbQ)TZj0CNai#r`k=}8iw?y(TJr75%tKM-xY7Y8>^de7Ee9PgNEE65&^})vw8-
%9^?c{JCYrAY8rf|$h9o;7lQ^+iJ+ZBbpeR;M8!$q*&d51<a!m}}3kJJBXZbO$kFRP4z1z!Z6o6~jGH=t&E0NJ9+J}}zbjA3m+JJ
Zs|2Ond(tH&7XNnM$5>H4T|Iv3PzQW)nWZ9K0^od!<M?(b{O9A+$n6up5o<6#Lo+yY`Y{fU>U%~68GF#5Vgz~&AN=TK!Pcqx9Gi6
Y~4H43l9MN#v;YNexd-+~~i5y}V#uo+ae1H7*Y--
>^mI|gm}qle~5X2hhfSiNh8^ZGV({p>kyw4a>109~CfO?&X?p3IM}d<tP>b6Rj1NI|WSK&xEgfs~gi1-
La$gJqYCjWvsLo{~WOx|#SS6@bO7(la2T<q*U@@f`_Lj?A}8FeJeQ3(>+Uq*}{x#5$W)Gf(ce{}jKZG`aw!=olnd_tev~h7R2+|D
C1;me^dS%9YDfqesZMuHTtWSv-xVKw!XL0}4L<0~0HN?<7>+A{A6oF(3%H%oW5-
4LWh^gKBv#WWfIQmNtyl%9J`W*2L9Up>pQxSt%<)>;%RXZwtf3^ycuonuS#$44uVqid`f6!p-
*pS53%A(2#G?4jidvW#{p_#Egmr0dAb`6R(eMsnU$*X@;`ofK4zgMiW_1?Ub6C=kt%Qd@Gn}(M0;eQ*^Y(2Z3r$DX&QJ)$DXeBb_
H@ssZn0`<8dnHuvCIwHg3W{a<i3{7^}+gay29OfO3d>gHP?CpPdR>cb2Y<lxzXcT%SXdxUiCilQ+S9q>_l!v}T}FNgAMu`H&nT_=
vtv<Y8|x^&`Jj;>rdupX;%Pk<;&HWEx6pf&|~qh1eTT2^dfqv7$$bB*KShEp7wc8V_@>NU4<Z#u!G0q2fX+K$)mF0?jnI*(%lIo7
T$ssY9X9!|wQtcYlPWzmnC-jOn!{=~=3Ot&ub75FqF4E;}vun;(O<6`-
SsQ?mjc%6k4uN8+?V7G@yYQ_!&*VRfD#~NAqXU5j8QAK+6Ff+eMQbEhbs*b-
Vt#iJTN9ZuyQE3&19IEUFa!y|K*QM9=?Dznb#ASnEXwg;vK<%^&GFSz~pLu$O*EMntL%H%07oF!@l!j(eLMGkjm@Z&Y7D}2qD^8Q
jiC76>nd)BNjk9JngzB5A`@vn|fvvHkzneF>i~1oQ&XvTVv>f+tgWcL1&;jC`jr@G*`!lC8hC57OW!C%aR7c-
KSmz1tWO4&CyookHOra}<2}B++E3R$)P{h&6wlW@Jm~eNq9)`(HVTg-
tIME6AjULz9B!EnynjltX1m5L*CwGMza=kfsb7_hVd5hsSA-IH%;<@YIhIb6<9<?8^QeCT$+mKv72l85O#;BMm931oUpV#D5gJA9
FbX-;M*s{*5?_LVSQZ!^tqVgz@k+(ss@Ov>yV|^lmXN!KNAm-N-;Z&H+*4<CkYrTffetGj2K%L~9qqjH`HWke!7}1@FX%QB0-
I5@&BhFwiygY~Gbi2LF+Y{`+5YUi3<YpR~%@FXEi#OBf&<~L$Os2vapxZJ~ku$=+n+V0Bb6T<aL~+ozdx(gEU}qxS$H2h=MB7}%9
Rwr9F;=FTk(I>mVff2OcQH7QRl3(gH6g65dJ+z?PdT>SsR<y7IeTs_o7S|>+5@(xosXG^I+sC*V~*pG8ANEFneQYB;vwDr^!)b_W
p_ZA&|^^NX0<O=)?OTV-Gg<q@Qpym63a1}<&x{eCc(o0OKKn&iuUqKS6tp9aHm+-
c_6292pEb<N+Lk~LQN4`NiS4Sj@>}zDirE&9T&M7D}hT;We+xEpaR%qM0I5DsR8p`5*~GVkDL{SIt)gjERhV%CEzcVBxD*m;2)M%
YgOH*<TP&yVDIJErLg?wnI{sMwn=#vqepX5H<kG?oBZ8$`vZp^M2T<;H=cYhHXqPSsT`(5C^gLu;7jyS!tMQ&pm{kt<(uS=H^#$J
S!98%mo1lB$xK$tj<f9NmEU&PW8r6oVX4ew_Vt~>seddr1r$l4w{`>2lYfG!ucU48Z7+n&ttLgLUoy)De1l!1=%TH09^FrY8<85o
?ezy+db?`JtuMoVNeHkwJ@S@s%E#3M@d8X*LxD|~gU?*(XuC6}LH_InLjPx&UMkd)#EM^(?)VAUik?!Mq&?+yT!vM4@E4j*KxTG8
jrwBy%?~5z=|Z1D4c6$CqbJmWKTe5)ifmu7r$M%Pu#&QuuGd_p7&NWCCUjukAx?J13##^?7Wf}}-VMKvYO2o&-
ThZ|CN?zG$$RHVDw026y1RJMemiV}Oq(mvW8+S}bg_Bgtr3~CS%In|DJyJig^Mv-
0|{o@H>T|+Bd?mf>je@D5gU&oVn{VN;K^&K@##eRX#A~&Qma`HYxibkwKMR;FVOE(;dkUM$f<x2#L$(ivX!xgcABoPY>|;aTu3wi
8;J&~N{&9%pr+MN1z0>{x^7<)cwfMKe82NpIOiB`8p|(T>&Cx&#i*L1M$yKhHv0rru-NEt>1<fnW-sP$q~9HT7$k5O%Ub&v7Cc;O
0#<#YZ}3KEWd`j_CIIp_4@xR^_oc>Oz?4<S-XJ|?e=}t?LFi!(V3*5?Z*N7LBU3NYt$1}#k#VoY@%)GM_(cBSB>nC*)G-v0iL6-
@C7CdLU0B|yG8k=6Vz!k<@y7~#{L1F0fhq&`MuQ_=u(mD=u$6?=>7&P1^M`xcJ;(}8llU7O<F|;NIx(<7;|wJ5pIbo~5~rEXI@Q^
$m4{^^aR!ZEPUN(Vdm%vk#~L2Lq#|O?42$L2Lgb9&(M+oWZIbcX7^&D6LR;`>mP1gLSB0q!P!YWMHOVCc!7MuN-=6+)<`X9j-
U$C7(86nF-AsLPDo>o!;qbq1TQ2WoTBqUVRvKdgE6`#@O(;2i7d9qAi?5=q?pf-OP3A3KL`gEp8g2#<(%G5Rt5j{l%~r9KZDq6k9
45nRAk@Hzo*)h}B1m_{Z{|xh3&)4*3NM2Cr(!jBy1(vbV&hUlUv96uW1-
dgQynLlCm_e$0PU+8y&X*Ks(U=*|Ai5@WhnTOZSV88PAqC{m>?ppYW?)i8USyYl!fWN@ZkJ&x3;&=@B>a(+DRncH(lyvQ0|k;1Za
>Cejs0DB_)=`jm_RAjt{bCjXy#@=Na92?K{uvxGoA*h7zrlfB&mL;RthGTy0ae4OT}z_^0I}tEm3wB5+6HhmqfrIc-
k&lp7;Dt9a&KLEsbApwV6<Qag;g5q=}O%3Uy&>k-fFw|#g;PrS0?+-?-S1k}}72;Jf2!I9dM{oQsI@%O#%ow7_yw-
o0Q_Wt1hJpcqNT#4#4Zqqp&Hnu!1=#PO-
r#%H9{Wj{6@A9S_e*SNaiE9@B4>r;eM#`qOQ9MF1!Qv|HMFO@JV?b5i<W4sUJS~%dZy+m=d-f&9v<Wn?6Lw`Y2hHl}gSwYm)FSS-
7_XnWWoOq$HV2RYb6TuuPVi93kRACJPNAAo^3BeFrAYzjy#`t_J{hA^F)>$ZsdIWG0yUO_4`-
Clk;55dxijnGirVoDM=ew|frpM8ZcJL!-
Sm4}ev%HtXAS_?%V3Uf%;R^ry&0H*fOgWC*z<U#UA%`s8?qsOW>ahhAT2>+r=ln(*C@j&rD5O7V<`@0QA`(aXo-
v4A}yDM)k4=S?%Sb3<%k-KK_*l_`ZMFeRK{&1qN!d>rD`}G6HC1P4`2LPMcBMJZt-
2|+O<S6N8K+~vIiUI5gTQHq<1SXyP{b?e=UeGFB#t<11r8-
{ji!LGOL`*yz*RB6fx@1Qn)Iu``Pj{$h0@ni0drZeRAOGffwXS9+DgSmev&(MQ4m%Mq|FPHa!snRGi`pdJhRZXkQL)6+GE56U#4W
Tu;f}jYTr2vLGKAmCko@fT-;coL}RPp>>{lK^gmZ&ZJljHNpTbA2U~+CdY-
1N^ZN8U}9dIsUD!z^E0TLrE`0L!@w@B)fvG=iIGBzx4kZKN~=UXr<81B^L7@ZZ^}HC<1D~_j1nBwD*xA)tFGZEleftks*W=+fL$X
=<=@{*A$;H=sBZFAiZ|e@M>dvpfih+TectuSCb(rzos{<-
kKRMP9MoK`3CGmLQqApbh;;gGD+v<6OY`IB@mj{i4UP4zQ!ev`KaFO@z{J>Dvpe)`dP0_zPa)6QiAc+L7x@~J<~sP0ceJ$#^|cw@
9Sh>{$Jnh<>f;edW6M7o+-
`iS5mp>NAUE8f9p=S=gzivbKPc8a1MX9TX3csenU^$Ob|&p_H8eeJ!eBf8?if(k?Y_;7WmhPIHljt#KeAGJwYSwE&#CW>7y5AWjf
jO86UQnC>0=SC={<Zk2XBfu;7ZaT>$>N0ax$}|2U_L)1KjWc8>{&pe){`jTT$aLN7eoZ5(2A6DT3TEnHS(b6)>;f2<k|7Sv}}uF%
lsbw*xPjq^~@s(WTcOP#F)^9>KjPf^}hWiJF+)GSjluy)r-
bp{<Hq6GOd>4%9|3NDFE7b}SO}`F~i7=J$MFkA$UjJxt$CJKZSYlhlul6Pm2i{SU%=TUZfw*57fxIp*mnNs3=ob@5k0Jf~4si86e
Jg7PZv=km)_f*NlGab@N*ctpr{0vr<#n5KBUa*=G^zG=U#{(DrhJfIslVpCU1WBBB?h+2#{m|$K;o<k?hJP3k7dzP%Omd*8oD_=n
m)>l8)%<jrknwkYyKB>Y>we$SQRR*vE!?Ec&STSnx2rN?l6I5(XaKBu`s$?KPSBOG=UM_zx3ZudgA=vyd2k`B>aXl`+=i5R9)0+D
k)xty`Day@m)B^%aEVeI_xlbDIY3L637iWw)h&5HOrOI1Vd?{B-b<&MN36mZi-
Qq&xIZA>Syvg{#`lxs8@iilyZy*VhJB=%w`AYTOf!QIZWei0~Q&;i-ztx4_R18ae7q--9W+0#$VS4*22U*yt!jd+xS)Az^Am<f2v
NM*gpJfo$B<$w^kn=80;pVUOiD(RGO}qZ>g{ZWSnDvZpEh%PZe@i#ryU9@(SVf#c)`u=3IkfGze#W;HodMocwdkxZlaBswxxZ2*U
E-
n!sLcnZ${x80g3@PUSnKbDm8J>06QT@^m`BVMaWA+|7QncqD>JmN2;eZHC%;5z3?Ui!ktXkrec@s)4eK}J4B1zwh0!vHa6SZ03U2
50#+SchRf{2*CfQKNVYQwMlIl9LD86<DQ)0FTi;1TrLSlue-cumg9|kPl!2N-k9{p9--$qzLCBG#=2V4#rJ?i)a@nR$IsvDd-fv+
Ih001qCHx}WZjZx2OmyDH$jP+bk9t2kHsbyXs2f;sMB8Q>um{x<dAKS!!vWv!^-C%DSv}CLz!U?SG#Rs-
i;C0gE4d`paoksvZZd+dp)02(csQ~1<oDm$fDy?^)ga4eApO+zO@wD&}hCPEa48sH{6hs|Uaj3Y~c?!J-
{wCTaO#G07>Dzy>qpc&hFe6dip-k-
v(+ZdzF=thoJddwPSC{Fe$ba1!8sV5vX&$n>6E)5zYB$#(lw*$P>Z;w2sh~Gcs(H2g#bzcF-
7i|#7o$GjoqPeb`uP4M=+5*%=39(YV_<9H_szZ;C;1jhE6;!9Oa{FKKZSD~s)M3s2riqOSs9Ydlgp5SN&k&k1-
p5XFxO;;G)_G4e^GLS4vCkxxl-H1>9F#kuaSbmTnDb5t`J-
=P?5TRV6lH8{)@d!N>eU$8_0WfPUcW|E2WFNwDaOr<K`5yv*JwqtUg-
|HA3ur4$hQFpAS`vGK6BycSLFX&qD{z6#rKpe>}quXF5MHsjUY+$_wSij$XCP8ch5Rlnd`4TAx6;MI;q!crh*j%<!Aza3{3Zl#7$
_`D*i4auyWWAZv=2C^8ABkqy+pGg-Hx(X4R;Yq>UbK2CQ|bf87I<W)jojCv3}$}G~Cd<o&5`wFS}6-
&)~Ozby@PzZEdU@_ntFA*e&j|eG#8UzjYTG8FpnG&;YtPmOz@{;!I-
vo{9hYd3LvgsO2&2WsOJlEsnuBIm~0n$U>o+@pnCJk>>4Fh*pN1Iw4sFT~{NKk>;M{Ax}9#Sm>!#cNjt8FvIl|)_I%BI4{s+eOrl
}<+yu4H(B{2W&fmvQ6uId{_jpKn6-
&<pR3e?<764{j$IMU1C_f`9JUNw;!@<rnV+y*RVRY)yZL+fdn)p$y3bJ<rWu%$k&tVLq4$I=98Cn7^Gg1-
fTo&t<>^9|TbFPS_W^c~SP;-HUTqsdg7WeUZvIthm?m5Hor$>wU^ng<+pm<gS+X%7$Oh7SqRW3*R@)B2_~}djum>{f(kE<gA-
c6k6b}#aJWN2LWUiiH={hBA>$oO;6u8wCGQ#S-
wJhtqkNB!OVPX1in83dm835o0NdC<C3a43Zvg`hm!w>EOyk+qkk=Vyo6J<w5wrU_MEPO8pYm?qr>#eAq^1A#C51`R<8iVFzV9cqe
5_OB;k{DK=usV#1-
?Z+9CrCaC>N=xm0gq9&e{=lS4O~N!l*D8i15uSR?(lPD`R_+8eUraoBOJYh+K|Ar&=SPu?<xrON+`V9J|+T_gRGT_qeGa>m-
*WjA2v?Vm27R;kY{zdBY1=A^_QQDc{Fav4lN3vMyjO+wc|N3+WwC7|M~lY-nHyhTw(fTX4)#Z<r41AwT@7`nu*eMgd?PMf$38;cB
~c;?;XzhFYYdW8e<QG?N+kI{#IfPXDvd)$SLxY>QzB&=^|P4e)DXBrR^$g1P-
=(GxF7IgtIZ~)jw^Klf^fieM}bbrJLoS13z<KS!nLF)=7PY7{+V79=PYa~+ksNt3W(hjeYO|qv<^WE;#8pk3(Ti;5!R!)+nF?Gox
u2VS=(x{S4&dX}eO#@E5I6llaga^OIZkF$uc`$S|9l?*kybhu$9T~9ETRv!>+=g@$P3j433N0hwOd*p2xTmh)v!?Swb0=i%zN6$}
|DsD#1(o3)8=ehxH4IBxX-sFN>mao?R~fL3bM$Io4UHQsXhZln9*$t(leb82f1^Sg={&W4RvS5D+tCAcINl5Wl?#~fnf{NkTdshX
`l!uNGGByyU==&-
1#n%rts|XkAw(vpesBBEO!ya8HTYbiy*)Hon1xnysPCO74cxDDFpa<j=~&*s<o&F0fC)K&QUrdqBE*TloH#`?wPWoS@!4E(?_oBG
AlUe)BxyvbqDpm)xh`i)N-nXckO_8<mg7vj?Wq=D#YBm66Ae=st}4jzT5;S62*DX-IN<hq)E>k4*4MO2H7^kx|2@3__++jog$3*z
xCc&MaiEHxjAAnvhlOe!96xXpW8J3+1Rzq#LfB3w{V3j+Srp2NsqKNVq{eV<HH;wGPqlul+S^+EGw@ouZBm0x=F&(v=AY8|g%#+o
Y^L;1;^S?=dW*pAwYjk=6Hj7CTk9-nX*bUNxGA!{_W;xAZ?m_SO&eu;k_3{rFYs;yF=DhYKy@R3aoty1@)f@FVVUc3HZdmM-ImCE
pCMt^yH4Z}|6@VQqsAKgxy$6p+>KC|f(n}{UGu_@okx<x9>+J>C=)iKg3rX)G&@In1XNMU2o@#16ZR946AaUrv8xQj$_u&53qCv3
ojdnJV_$V$^<OwSmaM6*`6j*e7$>Af{mLa{&*2?A?3(tW@jZ()53;&LfT9@_Rc(L#L!=tT8aeaw-
@u!g)#KQn_*}7k`NYG?oD=2hK#<SNwEOjVG>X>J_ygB2tfjHGUtXtN8ES=vI&x##3k;Y%I=*qPQ>n{StR5-F66h-_i?*-
<cuMzLVb#k!88t&@Z*P{J394(^B0oyS*evu(>U6MSBE69_?Nf>#sD;b(SwOS38Eye42kdM5(O<Tc^3cx$%F98EuYb`pngs)(cW+P
a<`|J@gg>ZB2ex-M5amtD{ed|=pJS#^qm^w?fdNx(#l_-
<zHlvwq;NZUjwW7Io(?14<i1}U?72uqQpQ<Pr?SBQWe9O?5EUH?AWsbiB}sfPfmk?W!;E&W6%rZ`eNIjwHb;iJdVY$HRqd?lkQLR
!G@&Nm=exmcXM6uUaRUATyNCT?JAucUIzR}<+e`#tv{u!R$J;;Rc~vNYqGwi{OQAd(t>b|S5`fddYb}UmakE1=3n(=w_h-Mdm%aI
Q7SIy)Wdr_R_RC63hFgOW5h~IdC9*;BG<S87MhF=)8c`Mm?IFi2mR7WtplkIket&}STUGB_l0qpavb--
%22XFE)cS^v3}aLIiO*J}Q_bL&_tv-y@oq?dO?f0;SUVRwHb$L-
Uk47oR6*AUC%ajCv$gR7Yo8P#glu<&Dz}iXu~%B&C8GbfyK9zLeLCp{`}B%LVA;PQm$*waMmxaOoBUZAl2P+6^F2CHAQmLb(aV#6
Fce>=)xoN-PXY1v9QQwVuw$*YW@UgvN40bo1e9X$x?n&U$Li#~Kg1SJiU}SLyc5e7RA?(BV{Ru1bScWEFzmN6)u^I>vdg^+ho{#V
3%miw8PY%db;b`Hng6%D^WsnJV2>Ii(VOPx*<1^wWYeBVFkE1jAPl5KZ)^)`-R|w-
O2ceZRZQB71~2emTtPd3R(+$l#w~%l@?LSlN+F*Mhrn^8{{X|`V#YAei^Lnm`8Qf=ss|kY0=OBEKD0Or>lw7jcL+9!RLdU&_@FN9
=@7F~9sagpeFGAC;m|s0c}@zd4aolfjx3bEj<!}_!@|68R{%`UZWhjAdKsj7Ezo)1=<GGLCPd?DykO$n_$HVMJ_9yY^K8T(MpiTA
Cdc@KPIo-GJm?oVJu+Ld8H%CgN-lFr;%j;Vj)Tn%BI99xeSdQhLcX}diU1<~`7I#j=g8GQ3P*2!4_L$L7`gi#__ijR&gU_Phl%F^
=MTl@o9#P3NBJ#8lhzul1~=!=){9Tjs?0OSZpZ<Sa;abU*?wAx7xyl$?vHGBRb^K%+XlMOtSU#+W5eH#k8UiWYxF%FS6Gs%)#zHR
GfBq{>-
m{mZj6ZXt{&)8VLT4o;!$eFRk)_4d~J;zpp7V*Jd;SJuUXK%JbXp5h@8CAl0qv;CuoXyvg(Nugj1lp31Af*6oE|!0pvlWGMv=mm&
qQ8JOx3MzCEtU<AxHS=z&eXPk3XTqq^s>x2?ff#)~XT*$V9<ExkK=b^*QJNwAu7%kc_x+rV|)IKL|=QkzO(hSjJG8Y<nUA~pm&8O
vxVMxay=`EjYBWX5QtgL@(w-G}7(itHq2w;+b`QK@1CQ<#U>aDX=^FoAISTsxND6fBSb0-U>QoLEzakC}-
yCKrj*qZZt*F2Eipps9-
ctCf`5Oi(&6(SS{&f)uXLhGPCM@!kVK=WP2(lGawy`1e3&YMluGgR?$U7CdK<Z;f0=7tqd}*&ImfK~}i5Z4h5pRebY1IlUiF+=v4
13}U}a@o6v-
Y?O9w3I(XfL(=5W{}?t5WQ{F@EhEFrjwMy(oi^{#Pau+x^`2bG+o3dVN=%2rpCa3&c`=3t(%M5)S;fv4UZB1t1;y>reKLuaR;ZVw
G$W(+7KR~CQ`O*V%Hv&zmHKVfZ(2-#4b;DHa2tlYA?n#)43K31$If%!{)KUcVny{<5w<j(r4LK{s?9Y;6C=9<NB(CSl-
I+D1wZ{JC2MlwcS~m2`oo_OSP||cb|Lypgb78`zSdW+v!D&G*Hn$=sRsaT{dc6U>G)mQb~#eC54a_DC(#E>hCG6TqDPsZz-
(Wt==iJ`;XNnNh;6oU;Wo}nqkbQVv3(hz{QJYw*`*@v90k)4?PTtT*EMz6_j?)Ye{%yeOEv)!v6|i-
Pfl_{7@9Y7l<_>7J|%JR@26ch!XyX#B`@gS%CXg=3lT@PHfxx!jjVGbMB(z+3)@<Jc~;X#1))sk*d)amsm9s2Q&}KWMV{K{hgNdR
3clhS*eB=bv_jv=yd+HSCSuC!ivVSu*?POwuN!Xesi;jj>GF<_j%*ix`i3cLgcc0QhKFk0SrgsU2wg~4FE!(zDRx(ShzQ8RCuC%9
h;>g)>?JcVbMnr#-
+R%^di~X(ur21yEs#c0q@s|~*>$e!#3sfqZVQ_^DY^bg)xYrYk=Tp~z+G;@e(RU`jPpW!{LibOf!`Ldgo924`f)a~AuOBPFljl!m
?TY#IZbe9_#{WqEgR<iK5mXPQ_&RKrN<@avYvpQyZV4AO*siZ%MYf3`$w%E7O{7K9hzau5O_U>by6}?iQ-TW3sQ0QV>p6jc~mZjw
j<o0Bu*^2Ni+NUt<uD&A3m&eO*;uWnScf@iXHBRjvdkKLAi6J3o(xRP6b+%BBI8|P8qTg7U#`xVMKr8aYD&CBc3LbYB7VUc)$QwC
acp=x#Hq`(m;zisV(6(!OT9qU|ixUc%>t78kKCbe5+@|TFupi#LSv!o4c#8uo$GU#W5LskyxW<wG}Su+Oi6r*^p+C$rCKUZL{Guw
KU2VMa$N*hQREv&gZ1IEjMbSISJX&KO=@BmWh2rhWGpmobD);8tV7XE$!xVvI^aI19Js4++lC1r~~w(Ozx36lXTF1t*T`ps?&0BL
$@1t8k<|`lZD|Bfarm%768pEXS79_Ffgghp(}<R(Eqz8n#=P~P@<!Nju%PzjWH31)YMlTv(exUw=SDd>PUewCNoI5YD_9Mo@&(f`
GmwCp}|BS0wg1&l1pNfMN+P8<ny{n`&T%l?DY|nW;UYXB5iwAMa4mP+ZTJllLDjrSOlTPHfpL)KA-
C9@xhu7O(gKMh|a)N5%YmIg3Ecjj@j<;z;CWSoxJ~zgP=2)1M~^s42MsfK6G9rT;#zI?G_gUMic^o<`G{uqU|x*kfh<Fcg_1IE#K
9N{cSVvM!EmXX4CXGy!JcqDVTOGjVcs8$im&4HDB9K9cFBF;-y+J=%>cG-RY(z;jP*12Cd8@N`ZacBJH)29r{$iCFNDv%hH?eeLB
nR;p}5<$(IaX26)_owRf29jstTCOSrOY)_X^Wxw|dU$)Y-
tUEcvD&Z(7ey@8Mghd6Td>jHTiAo#Le<ICMcJz$kGHdZkFFgZY3jrzqzb-49Gh6?z3XNQm-jI4gWWQHgM7`(tDFvzQLQQ+3o0ydq
xwKQY7bmxlC8M>nXC8Q9_FO2Zzym{VKjlPt1$O>Z$GIr&P5irrgkj>#}CLNrBdm7YC6Yo`j3+w9|#;Pr&O_;it#dGCmi};f?o1M1
FF9iW;4s;GnYlZ|MN?+oTVdQRf`nF_q@K<V(A-
vbkLWBpvM?%UI(A4<lsqz|U<ALFEi3i9lTXmW@2hnRsFQ!U@BY@Vtb;2x>ZW{c;hS<Ar;ZgS*sL-
bT9V6$T>_er;KwQdhqc@JM>;}X7Hh$sqo8{<Ch20G;bE-
ove-#Er9OWw*W8?iH>J37ssR>n8mzWui34HX%CMs==MD4oalz>w4q3^jN2o9&i(~481x+ig}eD|;Lj-gficJRqoZ9F)$bi)w4Cw~
C^oN@XoYneQp*S)hL?W3IcVtHzuZ~5(yoI*`a+nq&YqTG=nkkQ4uQhSxlaA8$hRZmf0J<t3?#8j|z3}OfsCFB1C5-
?Iu0}x28Qe>k+YB&&Y*h!k&BS9l-yUnWcge`y!>>4pUw0XEoy~snlYers$I)dmd%1FHsK6W?QbHJMF+hiys^$PK`#;m{(;x-
IQp_&GnSXMtfQCc|i5R#A)HjQ{QI=nLm-s07ezF~YS0nG6TWG%sm-~rtg6RInRDh$N)|2boesi@n)ELh*;c@XNx=(=4*SuM-
$%r6JcvpvY4_0~FuyubFDd`bokrhkl}txgh@@hYjBQDI19X&%r;1a)<6{uGbNR3s6ZF_=YSZhd%D62I4IURhzC_1U>~Kb3_3Dpjg
lc?QvrQrqD8o-{n9xlpFDi6B^(-?u+(|44E5It-rb*>D(r`v5O5RhQQY6NpFY$GZr*fS8#%PnFc440V~#&@oZnPa1sC+=)t&TZDF
RbT>$Z6oA)j!j)afuGT<W^U=*<M~7rRZ^oyHyYU6ae|4S#lH4t}m#<jkoEochW=+O1N{lx%U4#N2#^;fYc0W?$LY3D>85_pFjgv{
?0l&yWu`zftKW-$Gfw$A$;iq+eNhi^MIm-cG*x`e%a^yOAFFvzJH4i>DS>PjQ;gw~7$udL8z3X(AN}80UxW98c3<d;d@bdEplK2w
c=#p%5rSpps3X@vFJIsklB^V+yGza7R36@zEDNd>sBq5Hi+P>xGc~D6jOv4J(MduWv>3D1}2mT?~so+jf;I9mb{w^6z%{-
Qf*UUHi=Z1Y-
FW4C?+{a(|Jnv}OOw~FKcc@IQN&peMO_U7u3$T5mv;v{Zqz85(OVH%7zGWWtX_#3iDBv0jb}zBwp6gTfVJcGiD4u_?*$MYhl}=aS
H@PV@*&{9EZmSjWfbJbTXF6}FPX>b2&INE`Px(SC!V(n*>fWm76;Z3+N7)Jih&KP(##v>65|KM~*z%?|l|F`gA)l;4H|mg?vk(mN
dEz|)(!fI`wQ@|#00E@P0+gW$0{LlivBYQl0ssI200dcD
"""
_SOURCE_COMMIT = "c" * 40
_RAW_SHA256 = "d" * 64
_IMAGE_ID = f"sha256:{'f' * 64}"
_CHECKPOINT_WEIGHTS = [
    {
        "bytes": 3_957_109_648,
        "file": "model-00001-of-00017.safetensors",
        "sha256": "52562b2ff97b61764260273e71bf5b4cf8a66f569399398f26dec0300fcf1316",
    },
    {
        "bytes": 3_900_791_760,
        "file": "model-00002-of-00017.safetensors",
        "sha256": "e26764b2c6878e3fb7198895fa833ec62838d84a19665e6abfbae43c6daf02b3",
    },
    {
        "bytes": 3_900_791_760,
        "file": "model-00003-of-00017.safetensors",
        "sha256": "6c5ba7bed9c52bc121e75cbe8a7be46936d0006cc80f42a6d5886ed40b4c2a62",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00004-of-00017.safetensors",
        "sha256": "f736f6ac4d8c30866107fb1185a05b3c3cfce9717720082f466fa44e691bcec8",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00005-of-00017.safetensors",
        "sha256": "a52ed375c083209c54d42ac510afeb1fbb5af4f193be2dc7d103f665a0f212d3",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00006-of-00017.safetensors",
        "sha256": "37fae28990b0e4a70228549d040c0393e87bee3820d59e58e47844974d8dff5b",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00007-of-00017.safetensors",
        "sha256": "37776006aeaba29eca8bc73b2b963fe3477e1c2e3f6a27cb9527be75b905e1bf",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00008-of-00017.safetensors",
        "sha256": "73e74e9129674fe330948005075d70ab4fa0b92b68fb220c5a693f9cea553730",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00009-of-00017.safetensors",
        "sha256": "a044b3602a01bd8ea62ff51badf9cc038ab1d73d97399480e6a55b4c86fa7fa6",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00010-of-00017.safetensors",
        "sha256": "9966612ba7ecfc2cd2e592fb95224b86d743271ff88e172cc272a4b26382aa75",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00011-of-00017.safetensors",
        "sha256": "e2a058a0ac7d4b992b731c29221ecfb4b76b8a48d9004d0e8a62ba44f699845c",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00012-of-00017.safetensors",
        "sha256": "58a1aa89093fea07325f787072a468e3482a470ff4b7fe5ead5f749683907c40",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00013-of-00017.safetensors",
        "sha256": "35f3381bab31a23370c37d922290aeecdf603418336058fb86fe42d8f51ac40c",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00014-of-00017.safetensors",
        "sha256": "8713b062ddc178acf5917610b7f4b64eede833b2ea4aa37bd562dcf2f3a3339d",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00015-of-00017.safetensors",
        "sha256": "bec439d23931821a236d8f62fa79deecf5551bd25602278aa2ae0ce432b378cf",
    },
    {
        "bytes": 3_900_791_800,
        "file": "model-00016-of-00017.safetensors",
        "sha256": "e569139fadd61fe7c8f9eb1c976d9a627cae48c57ddf228cfbd0593c59c64ff7",
    },
    {
        "bytes": 3_055_341_992,
        "file": "model-00017-of-00017.safetensors",
        "sha256": "1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38",
    },
]


def _trace() -> dict[str, object]:
    selected = json.loads(
        lzma.decompress(
            base64.b85decode("".join(_TRACE_ROWS_B85.split()).encode())
        )
    )
    assert a8._canonical_digest(selected["scaling"]) == (
        a8.SCALING_SELECTION_SHA256
    )
    assert a8._canonical_digest(selected["affinity"]) == (
        a8.AFFINITY_SELECTION_SHA256
    )
    return {
        "bundle_sha256": a8.TRACE_BUNDLE_SHA256,
        "bundle_schema_version": a8.TRACE_SCHEMA,
        "sharegpt_descriptor_sha256": a8.SHAREGPT_DESCRIPTOR_SHA256,
        "shared_prefix_descriptor_sha256": a8.AFFINITY_DESCRIPTOR_SHA256,
        "scaling_selection_sha256": a8.SCALING_SELECTION_SHA256,
        "affinity_selection_sha256": a8.AFFINITY_SELECTION_SHA256,
        "scaling": selected["scaling"],
        "affinity": selected["affinity"],
    }


def _request(
    *,
    phase: str,
    arm: str,
    sequence: int,
    scheduled_ns: int,
    ended_ns: int,
    prompt_sha256: str,
    response_request_id: str | None,
    completion_tokens: int,
    repeat: int | None = None,
    rate_rps: int | None = None,
    prompt_index: int | None = None,
    session: int | None = None,
    turn: int | None = None,
    prompt_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> dict[str, object]:
    started_ns = scheduled_ns + 1_000
    first_token_ns = started_ns + 1_000_000
    row: dict[str, object] = {
        "type": "request",
        "phase": phase,
        "arm": arm,
        "retry_count": 0,
        "status": "success",
        "http_status": 200,
        "error_type": None,
        "saw_done": True,
        "finish_reason": "length",
        "sequence": sequence,
        "scheduled_ns": scheduled_ns,
        "started_ns": started_ns,
        "first_token_ns": first_token_ns,
        "ended_ns": ended_ns,
        "ttft_ns": first_token_ns - started_ns,
        "total_ns": ended_ns - started_ns,
        "output_sha256": f"{scheduled_ns:064x}",
        "response_request_id": response_request_id,
        "prompt_sha256": prompt_sha256,
        "completion_tokens": completion_tokens,
    }
    if repeat is not None:
        row["repeat"] = repeat
    if rate_rps is not None:
        row["rate_rps"] = rate_rps
    if prompt_index is not None:
        row["prompt_index"] = prompt_index
    if session is not None:
        row["session"] = session
    if turn is not None:
        row["turn"] = turn
    if prompt_tokens is not None:
        row["prompt_tokens"] = prompt_tokens
    if cached_tokens is not None:
        row["cached_tokens"] = cached_tokens
    row["namespace_sha256"] = a8._expected_namespace_sha256(row)
    row["namespace_token_id"] = a8._expected_namespace_token_id(row)
    return row


def _placement(
    request: dict[str, object],
    *,
    replica_id: str,
    reason: str,
) -> dict[str, object]:
    started_ns = int(request["started_ns"])
    latency_ns = 1_000_000
    row = {
        "type": "placement",
        "phase": request["phase"],
        "response_request_id": request["response_request_id"],
        "placement_started_ns": started_ns,
        "selected_at_ns": started_ns + latency_ns,
        "placement_latency_ns": latency_ns,
        "pool_size": 2,
        "eligible_size": 2,
        "replica_id": replica_id,
        "replica_generation": (
            "1" * 32 if replica_id == "0" else "2" * 32
        ),
        "reason": reason,
    }
    if request["phase"] == "affinity":
        row["session_sha256"] = a8.sha256_bytes(
            f"g2-a8-{int(request['session']):02d}".encode()
        )
    return row


def _append_request(
    rows: list[dict[str, object]],
    request: dict[str, object],
    *,
    replica_id: str = "0",
    reason: str = "least_work",
) -> None:
    rows.append(request)
    if request["arm"] == "dp":
        rows.append(
            _placement(
                request,
                replica_id=replica_id,
                reason=reason,
            )
        )


def _backends(role: str) -> dict[str, object]:
    engine: dict[str, object] = {
        "model": a8.MODEL,
        "engine_backend": (
            "kairyu" if role == "engine-host" else "replica-pool"
        ),
    }
    if role == "engine-host":
        engine["tensor_parallel_size"] = 4
    else:
        engine["via_replica"] = {"tensor_parallel_size": 4}
    return {"role": role, "engines": [engine]}


def _gpu_inventory() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "uuid": f"GPU-{index:064x}",
            "pci_bus_id": f"0000:{index + 1:02x}:00.0",
            "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "memory_total_mib": 97_887,
            "memory_used_mib": 40_000,
        }
        for index in range(8)
    ]


def _mount(
    destination: str,
    *,
    source: str,
    mount_type: str = "bind",
    name: str | None = None,
    rw: bool = False,
) -> dict[str, object]:
    return {
        "type": mount_type,
        "name": name,
        "source": source,
        "destination": destination,
        "rw": rw,
    }


def _container_provenance() -> dict[str, object]:
    expected_devices = {
        "replica0": ["0", "1", "2", "3"],
        "replica1": ["4", "5", "6", "7"],
        "gateway": [],
    }
    expected_cpus = {
        "replica0": "0-14,16-30,64-78,80-94",
        "replica1": "32-46,48-62,96-110,112-126",
        "gateway": "15,31,47,63,79,95,111,127",
    }
    expected_ports = {
        "replica0": "8200",
        "replica1": "8201",
        "gateway": "8202",
    }
    source_path = "repo:kairyu"
    config_paths = {
        "replica0": (
            "repo:examples/qwen3-32b-multi-gpu/a8-replica.yaml"
        ),
        "replica1": (
            "repo:examples/qwen3-32b-multi-gpu/a8-replica.yaml"
        ),
        "gateway": (
            "repo:examples/qwen3-32b-multi-gpu/a8-gateway.yaml"
        ),
    }
    services: dict[str, object] = {}
    for index, service in enumerate(("replica0", "replica1", "gateway")):
        mounts = [
            _mount("/app/kairyu", source=source_path),
            _mount(
                "/etc/kairyu/config.yaml",
                source=config_paths[service],
            ),
        ]
        if service != "gateway":
            mounts.append(
                _mount(
                    "/models",
                    source="volume:kairyu-qwen3-32b_qwen3-32b",
                    mount_type="volume",
                    name="kairyu-qwen3-32b_qwen3-32b",
                )
            )
        else:
            mounts.append(
                _mount(
                    "/evidence",
                    source="artifact:run-dir",
                    rw=True,
                )
            )
        services[service] = {
            "container_id": f"{index + 1:064x}",
            "image_id": _IMAGE_ID,
            "running": True,
            "started_at": f"2026-07-31T00:00:0{index}Z",
            "compose_config_sha256": "e" * 64,
            "device_ids": expected_devices[service],
            "cpuset_cpus": expected_cpus[service],
            "port": expected_ports[service],
            "mounts": mounts,
        }
    return {
        "project": "kairyu-qwen3-32b-a8",
        "compose_file": (
            "repo:examples/qwen3-32b-multi-gpu/a8-compose.yaml"
        ),
        "services": services,
    }


def _provenance() -> dict[str, object]:
    containers = _container_provenance()
    replica0 = containers["services"]["replica0"]
    return {
        "source_commit": _SOURCE_COMMIT,
        "tracked_tree_clean": True,
        "image_id": _IMAGE_ID,
        "model": a8.MODEL,
        "model_revision": a8.MODEL_REVISION,
        "gpu_count": 8,
        "gpus": _gpu_inventory(),
        "nvidia_smi_topology": [
            *[
                f"GPU{index} X PIX PIX PIX PIX PIX PIX PIX"
                for index in range(8)
            ],
        ],
        "single_gpu_ordinals": [0, 1, 2, 3],
        "dp_gpu_ordinals": list(range(8)),
        "single_url": "http://127.0.0.1:8200/v1",
        "dp_url": "http://127.0.0.1:8202/v1",
        "single_backends": _backends("engine-host"),
        "dp_backends": _backends("gateway"),
        "checkpoint": {
            "model": a8.MODEL,
            "revision": a8.MODEL_REVISION,
            "weights_sha256": a8.CHECKPOINT_WEIGHTS_SHA256,
            "index_sha256": a8.CHECKPOINT_INDEX_SHA256,
            "checkpoint_evidence_sha256": (
                a8.CHECKPOINT_EVIDENCE_SHA256
            ),
            "tokenizer_sha256": a8.PINNED_TOKENIZER_SHA256,
            "a7_raw_sha256": a8.A7_RAW_SHA256,
            "a7_manifest_sha256": a8.A7_MANIFEST_SHA256,
            "attested_container_id": replica0["container_id"],
            "live_volume_attestation": {
                "weights": copy.deepcopy(_CHECKPOINT_WEIGHTS),
                "weights_sha256": a8.CHECKPOINT_WEIGHTS_SHA256,
                "index_sha256": a8.CHECKPOINT_INDEX_SHA256,
                "tokenizer_sha256": a8.PINNED_TOKENIZER_SHA256,
                "config_sha256": a8.CHECKPOINT_CONFIG_SHA256,
            },
        },
        "config_file_sha256": a8._config_file_hashes(),
        "containers": containers,
    }


def _passing_rows() -> list[dict[str, object]]:
    trace = _trace()
    scaling_trace = trace["scaling"]
    affinity_trace = trace["affinity"]
    assert isinstance(scaling_trace, list)
    assert isinstance(affinity_trace, list)
    provenance = _provenance()
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": a8.SCHEMA_VERSION,
            "gate": a8.GATE,
            "config": a8.formal_config(),
            "trace": trace,
            "provenance": provenance,
            "started_unix_ns": 1,
            "started_monotonic_ns": 1,
        }
    ]
    clock_ns = 1_000_000_000_000
    request_number = 0

    for repeat, arm_order in enumerate(a8.ARM_ORDERS):
        for arm in arm_order:
            for sequence in range(a8.WARMUP_REQUESTS):
                scheduled_ns = clock_ns + sequence * 2_000_000
                request_id = (
                    f"warmup-r{repeat}-{arm}-{sequence}"
                    if arm == "dp"
                    else None
                )
                request = _request(
                    phase="warmup",
                    arm=arm,
                    repeat=repeat,
                    sequence=sequence,
                    scheduled_ns=scheduled_ns,
                    ended_ns=scheduled_ns + 2_000_000,
                    prompt_index=sequence,
                    prompt_sha256=str(scaling_trace[sequence]["prompt_sha256"]),
                    response_request_id=request_id,
                    completion_tokens=a8.SCALING_OUTPUT_TOKENS,
                )
                _append_request(
                    rows,
                    request,
                    replica_id=str(sequence % 2),
                )
                request_number += 1
            clock_ns += 100_000_000

            for rate_rps in a8.RATES_RPS:
                cell_start_ns = clock_ns
                target_wall_ns = (
                    1_900_000_000 if arm == "single" else 1_000_000_000
                )
                for sequence in range(a8.SCALING_REQUESTS):
                    scheduled_ns = (
                        cell_start_ns
                        + round(sequence * 1_000_000_000 / rate_rps)
                    )
                    if sequence == a8.SCALING_REQUESTS - 1:
                        ended_ns = cell_start_ns + max(
                            target_wall_ns,
                            scheduled_ns - cell_start_ns + 2_000_000,
                        )
                    else:
                        ended_ns = scheduled_ns + 2_000_000
                    request_id = (
                        f"scaling-r{repeat}-{arm}-{rate_rps}-{sequence}"
                        if arm == "dp"
                        else None
                    )
                    prompt_index = a8._scaling_prompt_index(repeat, sequence)
                    request = _request(
                        phase="scaling",
                        arm=arm,
                        repeat=repeat,
                        rate_rps=rate_rps,
                        sequence=sequence,
                        scheduled_ns=scheduled_ns,
                        ended_ns=ended_ns,
                        prompt_index=prompt_index,
                        prompt_sha256=str(
                            scaling_trace[prompt_index]["prompt_sha256"]
                        ),
                        response_request_id=request_id,
                        completion_tokens=a8.SCALING_OUTPUT_TOKENS,
                    )
                    _append_request(
                        rows,
                        request,
                        replica_id=str(sequence % 2),
                    )
                    request_number += 1
                clock_ns += 30_000_000_000

    for arm in ("single", "dp"):
        for sequence in range(a8.AFFINITY_REQUESTS):
            trace_row = affinity_trace[sequence]
            assert isinstance(trace_row, dict)
            scheduled_ns = clock_ns + sequence * 2_000_000
            request_id = f"affinity-{arm}-{sequence}" if arm == "dp" else None
            request = _request(
                phase="affinity",
                arm=arm,
                sequence=sequence,
                scheduled_ns=scheduled_ns,
                ended_ns=scheduled_ns + 2_000_000,
                prompt_sha256=str(trace_row["prompt_sha256"]),
                response_request_id=request_id,
                completion_tokens=a8.AFFINITY_OUTPUT_TOKENS,
                session=int(trace_row["session"]),
                turn=int(trace_row["turn"]),
                prompt_tokens=100,
                cached_tokens=90 if arm == "dp" else 100,
            )
            _append_request(
                rows,
                request,
                replica_id=str(int(trace_row["session"]) % 2),
                reason="session_affinity",
            )
            request_number += 1
        clock_ns += 2_000_000_000

    rows.append(
        {
            "type": "run_end",
            "source_commit": _SOURCE_COMMIT,
            "completed": True,
            "request_count": request_number,
            "placement_count": sum(
                row.get("type") == "placement" for row in rows
            ),
            "containers": copy.deepcopy(provenance["containers"]),
            "single_backends": copy.deepcopy(
                provenance["single_backends"]
            ),
            "dp_backends": copy.deepcopy(provenance["dp_backends"]),
            "gpu_inventory": copy.deepcopy(provenance["gpus"]),
            "ended_unix_ns": 2,
            "ended_monotonic_ns": 10**18,
        }
    )
    return rows


def _row(
    rows: list[dict[str, object]],
    **criteria: object,
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    assert len(matches) == 1, (criteria, len(matches))
    return matches[0]


def _mount_row(
    rows: list[dict[str, object]],
    *,
    service: str,
    destination: str,
) -> dict[str, object]:
    mounts = rows[0]["provenance"]["containers"]["services"][service][
        "mounts"
    ]
    matches = [
        mount
        for mount in mounts
        if mount.get("destination") == destination
    ]
    assert len(matches) == 1, (service, destination, len(matches))
    return matches[0]


def _rewrite_raw(
    artifact: Path,
    mutation: Callable[[list[dict[str, object]]], None],
) -> None:
    raw_path = artifact / a8.RAW_NAME
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    mutation(rows)
    a8._write_jsonl(raw_path, rows)


def _change_rate_outside_grid(rows: list[dict[str, object]]) -> None:
    request = _row(
        rows,
        type="request",
        phase="scaling",
        repeat=0,
        arm="single",
        rate_rps=4,
        sequence=0,
    )
    request["rate_rps"] = 3


def _break_open_loop_cadence(rows: list[dict[str, object]]) -> None:
    request = _row(
        rows,
        type="request",
        phase="scaling",
        repeat=0,
        arm="single",
        rate_rps=4,
        sequence=1,
    )
    for field in (
        "scheduled_ns",
        "started_ns",
        "first_token_ns",
        "ended_ns",
    ):
        request[field] = int(request[field]) + 1


def _migrate_affinity_session(rows: list[dict[str, object]]) -> None:
    placement = _row(
        rows,
        type="placement",
        phase="affinity",
        response_request_id="affinity-dp-0",
    )
    replica_id = "1" if placement["replica_id"] == "0" else "0"
    placement["replica_id"] = replica_id
    placement["replica_generation"] = (
        "1" * 32 if replica_id == "0" else "2" * 32
    )


def _escape_placement_request_bounds(rows: list[dict[str, object]]) -> None:
    placement = _row(
        rows,
        type="placement",
        phase="scaling",
        response_request_id="scaling-r0-dp-4-0",
    )
    request = _row(
        rows,
        type="request",
        phase="scaling",
        response_request_id="scaling-r0-dp-4-0",
    )
    placement["placement_started_ns"] = int(request["started_ns"]) - 1
    placement["selected_at_ns"] = int(request["first_token_ns"])
    placement["placement_latency_ns"] = (
        int(placement["selected_at_ns"])
        - int(placement["placement_started_ns"])
    )


@pytest.fixture(scope="module")
def passing_rows() -> list[dict[str, object]]:
    return _passing_rows()


@pytest.fixture(autouse=True)
def pin_committed_config_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = a8._config_file_hashes()
    monkeypatch.setattr(
        a8,
        "_committed_config_file_hashes",
        lambda _commit: expected,
    )


def test_valid_synthetic_artifact_passes_all_binding_checks(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    artifact = tmp_path / "artifact"
    manifest = a8.seal_artifact(artifact, copy.deepcopy(passing_rows))

    assert manifest["passed"] is True
    assert all(manifest["checks"].values())
    assert manifest["goodput"]["round_dp_over_single_ratios"] == pytest.approx(
        [1.9, 1.9, 1.9]
    )
    assert manifest["router"]["placement_latency_p99_ns"] == 1_000_000
    assert manifest["affinity"]["dp_over_single_cache_rate"] == pytest.approx(
        0.9
    )
    assert a8.verify_artifact(artifact, assert_gate=True) == manifest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows[0]["config"].__setitem__("repeats", 2),
            "formal config differs",
        ),
        (
            _change_rate_outside_grid,
            "scaling namespace rate invalid",
        ),
        (
            _break_open_loop_cadence,
            "fixed open-loop cadence",
        ),
        (
            lambda rows: _row(
                rows,
                type="request",
                phase="scaling",
                repeat=0,
                arm="single",
                rate_rps=4,
                sequence=0,
            ).__setitem__("completion_tokens", 31),
            "exactly 32 tokens",
        ),
        (
            lambda rows: _row(
                rows,
                type="request",
                phase="scaling",
                repeat=0,
                arm="single",
                rate_rps=4,
                sequence=0,
            ).__setitem__("output_sha256", ""),
            "output digest is invalid",
        ),
        (
            lambda rows: _row(
                rows,
                type="placement",
                phase="scaling",
                response_request_id="scaling-r0-dp-4-0",
            ).__setitem__("response_request_id", "orphan"),
            "does not join",
        ),
        (
            _escape_placement_request_bounds,
            "placement timestamps escape the joined client request bounds",
        ),
        (
            lambda rows: _row(
                rows,
                type="placement",
                phase="scaling",
                response_request_id="scaling-r0-dp-4-0",
            ).__setitem__("replica_generation", ""),
            "placement replica generation is invalid",
        ),
        (
            lambda rows: _row(
                rows,
                type="placement",
                phase="scaling",
                response_request_id="scaling-r0-dp-4-2",
            ).__setitem__("replica_generation", "3" * 32),
            "replica generation changed during the formal measurement",
        ),
        (
            lambda rows: _row(
                rows,
                type="request",
                phase="scaling",
                repeat=0,
                arm="single",
                rate_rps=4,
                sequence=0,
            ).__setitem__("ttft_ns", 1_000_001),
            "TTFT does not match",
        ),
        (
            lambda rows: _row(
                rows,
                type="request",
                phase="affinity",
                arm="dp",
                sequence=0,
            ).__setitem__("cached_tokens", 101),
            "exceeds prompt_tokens",
        ),
        (
            _migrate_affinity_session,
            "migrated a conversation",
        ),
        (
            lambda rows: _row(
                rows,
                type="placement",
                phase="affinity",
                response_request_id="affinity-dp-0",
            ).__setitem__("session_sha256", "0" * 64),
            "session hash does not match",
        ),
        (
            lambda rows: rows[0]["trace"]["affinity"][0].__setitem__(
                "prompt_tokens",
                639,
            ),
            "affinity trace geometry or identity",
        ),
        (
            lambda rows: rows[0]["provenance"].__setitem__(
                "source_commit",
                "c" * 64,
            ),
            "source identity is invalid or dirty",
        ),
    ],
    ids=[
        "config",
        "arrival-grid",
        "arrival-cadence",
        "output-length",
        "output-digest",
        "placement-join",
        "placement-request-bounds",
        "placement-generation-format",
        "placement-generation-stability",
        "timestamps",
        "cache-usage",
        "session-affinity",
        "session-header-hash",
        "affinity-trace-geometry",
        "source-commit",
    ],
)
def test_semantic_raw_tampering_is_rejected_during_verification(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
    mutation: Callable[[list[dict[str, object]]], None],
    message: str,
) -> None:
    artifact = tmp_path / "artifact"
    a8.seal_artifact(artifact, copy.deepcopy(passing_rows))
    _rewrite_raw(artifact, mutation)

    with pytest.raises(a8.GateEvidenceError, match=message):
        a8.verify_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows[0]["provenance"]["checkpoint"].__setitem__(
                "weights_sha256",
                "0" * 64,
            ),
            "checkpoint provenance differs",
        ),
        (
            lambda rows: rows[0]["provenance"]["checkpoint"].__setitem__(
                "attested_container_id",
                "9" * 64,
            ),
            "checkpoint provenance differs",
        ),
        (
            lambda rows: rows[0]["provenance"]["checkpoint"][
                "live_volume_attestation"
            ]["weights"].pop(),
            "live checkpoint shard evidence is invalid",
        ),
        (
            lambda rows: rows[0]["provenance"]["checkpoint"][
                "live_volume_attestation"
            ]["weights"][0].__setitem__("sha256", "0" * 64),
            "live model-volume bytes differ from the pinned checkpoint",
        ),
        (
            lambda rows: rows[0]["provenance"]["checkpoint"][
                "live_volume_attestation"
            ].__setitem__("index_sha256", "0" * 64),
            "live model-volume bytes differ from the pinned checkpoint",
        ),
        (
            lambda rows: rows[0]["provenance"]["checkpoint"][
                "live_volume_attestation"
            ].__setitem__("tokenizer_sha256", "0" * 64),
            "live model-volume bytes differ from the pinned checkpoint",
        ),
        (
            lambda rows: rows[0]["provenance"]["checkpoint"][
                "live_volume_attestation"
            ].__setitem__("config_sha256", "0" * 64),
            "live model-volume bytes differ from the pinned checkpoint",
        ),
        (
            lambda rows: rows[0]["provenance"]["single_backends"][
                "engines"
            ][0].__setitem__("tensor_parallel_size", 8),
            "does not prove one Kairyu TP4 engine",
        ),
        (
            lambda rows: rows[0]["provenance"]["gpus"][1].__setitem__(
                "uuid",
                rows[0]["provenance"]["gpus"][0]["uuid"],
            ),
            "GPU UUID/PCI identity is not unique",
        ),
        (
            lambda rows: rows[0]["provenance"]["containers"]["services"][
                "replica1"
            ].__setitem__("device_ids", ["0", "1", "2", "3"]),
            "replica1 runtime identity mismatch",
        ),
        (
            lambda rows: rows[0]["provenance"]["containers"]["services"][
                "replica0"
            ].__setitem__("started_at", ""),
            "replica0 runtime identity mismatch",
        ),
        (
            lambda rows: rows[0]["provenance"]["containers"]["services"][
                "replica0"
            ].__setitem__("compose_config_sha256", "invalid"),
            "replica0 runtime identity mismatch",
        ),
        (
            lambda rows: _mount_row(
                rows,
                service="replica0",
                destination="/app/kairyu",
            ).__setitem__("source", "repo:different"),
            "replica0 source/config mounts are not read-only",
        ),
        (
            lambda rows: _mount_row(
                rows,
                service="replica0",
                destination="/etc/kairyu/config.yaml",
            ).__setitem__("source", "repo:different"),
            "replica0 source/config mounts are not read-only",
        ),
        (
            lambda rows: _mount_row(
                rows,
                service="replica0",
                destination="/models",
            ).__setitem__("source", "volume:different"),
            "replica0 model volume identity mismatch",
        ),
        (
            lambda rows: _mount_row(
                rows,
                service="gateway",
                destination="/evidence",
            ).__setitem__("source", "artifact:different"),
            "gateway evidence mount identity mismatch",
        ),
        (
            lambda rows: rows[0]["provenance"][
                "config_file_sha256"
            ].pop("examples/qwen3-32b-multi-gpu/a8-stack.sh"),
            "formal config source hashes are incomplete",
        ),
    ],
    ids=[
        "checkpoint",
        "checkpoint-container-binding",
        "checkpoint-shard-count",
        "checkpoint-shard-bytes",
        "checkpoint-index",
        "checkpoint-tokenizer",
        "checkpoint-config",
        "backend-topology",
        "gpu-identity",
        "container-partition",
        "container-started-at",
        "container-compose-config",
        "container-source-mount",
        "container-config-mount",
        "container-model-volume",
        "container-evidence-mount",
        "config-source",
    ],
)
def test_provenance_tampering_is_rejected(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
    mutation: Callable[[list[dict[str, object]]], None],
    message: str,
) -> None:
    artifact = tmp_path / "artifact"
    a8.seal_artifact(artifact, copy.deepcopy(passing_rows))
    _rewrite_raw(artifact, mutation)

    with pytest.raises(a8.GateEvidenceError, match=message):
        a8.verify_artifact(artifact)


def test_all_38_cache_namespaces_are_one_token_and_arm_disjoint(
    passing_rows: list[dict[str, object]],
) -> None:
    requests = [
        row for row in passing_rows if row.get("type") == "request"
    ]
    namespaces: dict[
        tuple[object, object, object, object],
        tuple[object, object],
    ] = {}
    for row in requests:
        identity = (
            row["phase"],
            row["arm"],
            row.get("repeat"),
            row.get("rate_rps"),
        )
        pair = (row["namespace_sha256"], row["namespace_token_id"])
        assert pair == (
            a8._expected_namespace_sha256(row),
            a8._expected_namespace_token_id(row),
        )
        namespaces.setdefault(identity, pair)
        assert namespaces[identity] == pair

    assert len(namespaces) == len(a8.NAMESPACE_PIECES) == 38
    assert len(set(namespaces.values())) == 38
    single = {
        pair
        for identity, pair in namespaces.items()
        if identity[1] == "single"
    }
    dp = {
        pair
        for identity, pair in namespaces.items()
        if identity[1] == "dp"
    }
    assert single.isdisjoint(dp)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("namespace_sha256", "request cache namespace mismatch"),
        ("namespace_token_id", "request cache namespace token ID mismatch"),
    ],
)
def test_cache_namespace_tampering_is_rejected(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
    field: str,
    message: str,
) -> None:
    artifact = tmp_path / "artifact"
    a8.seal_artifact(artifact, copy.deepcopy(passing_rows))

    def mutate(rows: list[dict[str, object]]) -> None:
        single = _row(
            rows,
            type="request",
            phase="scaling",
            repeat=0,
            arm="single",
            rate_rps=4,
            sequence=0,
        )
        dp = _row(
            rows,
            type="request",
            phase="scaling",
            repeat=0,
            arm="dp",
            rate_rps=4,
            sequence=0,
        )
        single[field] = dp[field]

    _rewrite_raw(artifact, mutate)
    with pytest.raises(a8.GateEvidenceError, match=message):
        a8.verify_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows[-1].__setitem__(
                "request_count",
                int(rows[-1]["request_count"]) + 1,
            ),
            "run_end request count mismatch",
        ),
        (
            lambda rows: rows[-1].__setitem__(
                "placement_count",
                int(rows[-1]["placement_count"]) - 1,
            ),
            "run_end placement count mismatch",
        ),
        (
            lambda rows: rows[-1]["containers"]["services"][
                "gateway"
            ].__setitem__("running", False),
            "container runtime changed",
        ),
        (
            lambda rows: rows[-1]["dp_backends"]["engines"][
                0
            ].__setitem__("engine_backend", "different"),
            "/backends topology changed",
        ),
        (
            lambda rows: rows[-1]["gpu_inventory"][0].__setitem__(
                "uuid",
                "GPU-changed",
            ),
            "physical GPU inventory changed",
        ),
        (
            lambda rows: rows[-1].__setitem__(
                "ended_monotonic_ns",
                2,
            ),
            "request timestamps escape the run bounds",
        ),
    ],
    ids=[
        "request-count",
        "placement-count",
        "container",
        "backends",
        "gpu-inventory",
        "run-bounds",
    ],
)
def test_run_end_tampering_is_rejected(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
    mutation: Callable[[list[dict[str, object]]], None],
    message: str,
) -> None:
    artifact = tmp_path / "artifact"
    a8.seal_artifact(artifact, copy.deepcopy(passing_rows))
    _rewrite_raw(artifact, mutation)

    with pytest.raises(a8.GateEvidenceError, match=message):
        a8.verify_artifact(artifact)


def test_paired_arm_order_tampering_is_rejected(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    artifact = tmp_path / "artifact"
    a8.seal_artifact(artifact, copy.deepcopy(passing_rows))

    def move_dp_before_single(rows: list[dict[str, object]]) -> None:
        moved = _row(
            rows,
            type="request",
            phase="scaling",
            repeat=0,
            arm="dp",
            rate_rps=4,
            sequence=0,
        )
        rows.remove(moved)
        rows.insert(1, moved)

    _rewrite_raw(artifact, move_dp_before_single)
    with pytest.raises(a8.GateEvidenceError, match="paired arm order"):
        a8.verify_artifact(artifact)


def test_goodput_below_floor_is_retained_as_a_failed_gate(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(passing_rows)
    for repeat in range(a8.REPEATS):
        first = _row(
            rows,
            type="request",
            phase="scaling",
            repeat=repeat,
            arm="dp",
            rate_rps=64,
            sequence=0,
        )
        last = _row(
            rows,
            type="request",
            phase="scaling",
            repeat=repeat,
            arm="dp",
            rate_rps=64,
            sequence=a8.SCALING_REQUESTS - 1,
        )
        ended_ns = int(first["scheduled_ns"]) + 1_000_000_001
        last["ended_ns"] = ended_ns
        last["total_ns"] = ended_ns - int(last["started_ns"])

    artifact = tmp_path / "artifact"
    manifest = a8.seal_artifact(artifact, rows)
    check = "median_peak_goodput_ratio_at_least_1_9"
    assert manifest["passed"] is False
    assert manifest["checks"][check] is False
    assert a8.verify_artifact(artifact)["passed"] is False
    with pytest.raises(a8.GateEvidenceError, match=check):
        a8.verify_artifact(artifact, assert_gate=True)


def test_zero_single_goodput_is_retained_as_an_explicit_failed_gate(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(passing_rows)
    for row in rows:
        if (
            row.get("type") == "request"
            and row.get("phase") == "scaling"
            and row.get("arm") == "single"
        ):
            first_token_ns = (
                int(row["started_ns"]) + a8.TTFT_SLO_NS + 1
            )
            row["first_token_ns"] = first_token_ns
            row["ttft_ns"] = first_token_ns - int(row["started_ns"])
            row["ended_ns"] = max(int(row["ended_ns"]), first_token_ns)
            row["total_ns"] = int(row["ended_ns"]) - int(
                row["started_ns"]
            )

    artifact = tmp_path / "artifact"
    manifest = a8.seal_artifact(artifact, rows)
    check = "single_peak_goodput_positive_each_repeat"
    assert manifest["passed"] is False
    assert manifest["checks"][check] is False
    assert manifest["goodput"]["round_dp_over_single_ratios"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert manifest["goodput"]["median_dp_over_single_ratio"] == 0.0
    assert a8.verify_artifact(artifact) == manifest


def test_router_p99_uses_nearest_rank_and_excludes_the_10ms_boundary(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(passing_rows)
    placements = [
        row
        for row in rows
        if row.get("type") == "placement" and row.get("phase") == "scaling"
    ]
    assert len(placements) == a8.REPEATS * len(a8.RATES_RPS) * a8.SCALING_REQUESTS
    requests_by_id = {
        row["response_request_id"]: row
        for row in rows
        if row.get("type") == "request" and row.get("arm") == "dp"
    }

    def set_placement_latency(
        placement: dict[str, object],
        latency_ns: int,
    ) -> None:
        placement["placement_latency_ns"] = latency_ns
        selected_ns = int(placement["placement_started_ns"]) + latency_ns
        placement["selected_at_ns"] = selected_ns
        request = requests_by_id[placement["response_request_id"]]
        request["first_token_ns"] = selected_ns
        request["ttft_ns"] = selected_ns - int(request["started_ns"])
        if int(request["ended_ns"]) < selected_ns:
            request["ended_ns"] = selected_ns
            request["total_ns"] = selected_ns - int(request["started_ns"])

    for row in placements[:9]:
        set_placement_latency(row, 50_000_000)

    below_rank = a8.recompute_manifest(rows, raw_sha256=_RAW_SHA256)
    assert below_rank["router"]["placement_latency_p99_ns"] == 1_000_000
    assert below_rank["checks"]["scaling_placement_latency_p99_below_10ms"]

    boundary = placements[9]
    set_placement_latency(boundary, a8.PLACEMENT_P99_CEILING_NS)
    artifact = tmp_path / "artifact"
    manifest = a8.seal_artifact(artifact, rows)
    check = "scaling_placement_latency_p99_below_10ms"
    assert manifest["router"]["placement_latency_p99_ns"] == 10_000_000
    assert manifest["checks"][check] is False
    with pytest.raises(a8.GateEvidenceError, match=check):
        a8.verify_artifact(artifact, assert_gate=True)


def test_cache_retention_below_floor_is_retained_as_a_failed_gate(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(passing_rows)
    for row in rows:
        if (
            row.get("type") == "request"
            and row.get("phase") == "affinity"
            and row.get("arm") == "dp"
        ):
            row["cached_tokens"] = 89

    artifact = tmp_path / "artifact"
    manifest = a8.seal_artifact(artifact, rows)
    check = "affinity_cache_rate_retention_at_least_0_90"
    assert manifest["affinity"]["dp_over_single_cache_rate"] == pytest.approx(
        0.89
    )
    assert manifest["checks"][check] is False
    with pytest.raises(a8.GateEvidenceError, match=check):
        a8.verify_artifact(artifact, assert_gate=True)


def test_zero_single_affinity_cache_rate_is_retained_as_failed_gate(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(passing_rows)
    for row in rows:
        if (
            row.get("type") == "request"
            and row.get("phase") == "affinity"
            and row.get("arm") == "single"
        ):
            row["cached_tokens"] = 0

    artifact = tmp_path / "artifact"
    manifest = a8.seal_artifact(artifact, rows)
    check = "single_affinity_cache_rate_positive"
    assert manifest["passed"] is False
    assert manifest["checks"][check] is False
    assert manifest["affinity"]["single"]["cache_rate"] == 0.0
    assert manifest["affinity"]["dp_over_single_cache_rate"] == 0.0
    assert a8.verify_artifact(artifact) == manifest


def test_raw_byte_and_derived_manifest_tampering_are_rejected(
    tmp_path: Path,
    passing_rows: list[dict[str, object]],
) -> None:
    raw_artifact = tmp_path / "raw"
    a8.seal_artifact(raw_artifact, copy.deepcopy(passing_rows))
    raw_path = raw_artifact / a8.RAW_NAME
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8").replace(
            '"type":"run"',
            '"type": "run"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(a8.GateEvidenceError, match="differs from raw replay"):
        a8.verify_artifact(raw_artifact)

    manifest_artifact = tmp_path / "manifest"
    manifest = a8.seal_artifact(
        manifest_artifact,
        copy.deepcopy(passing_rows),
    )
    manifest["goodput"]["median_dp_over_single_ratio"] = 999.0
    (manifest_artifact / a8.MANIFEST_NAME).write_text(
        f"{a8.canonical_json(manifest)}\n",
        encoding="utf-8",
    )
    with pytest.raises(a8.GateEvidenceError, match="differs from raw replay"):
        a8.verify_artifact(manifest_artifact)


def test_help_is_side_effect_free(tmp_path: Path) -> None:
    script = Path(a8.__file__).resolve()
    before = list(tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert list(tmp_path.iterdir()) == before
