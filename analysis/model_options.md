# Modeller att diskutera på mötet

Mitt förslag är att behålla XGBoost som jämförelse och sedan testa **Deep Sets/PFN, ParticleNet-Lite och en liten Particle Transformer**.
Det här är förslag för vårt BIB-problem. Artiklarna visar resultat för andra typer av jetklassificering.

## Vad jag tog från materialet

- `2603.12306v1.pdf`, s. 1-3: en översikt över nätverk som använder jetens beståndsdelar. Den tar upp PFN/EFN, ParticleNet, ParT och ATLAS-varianten DeParT.
- `QT_TGM.pdf`, del I-II, särskilt s. 24-32: BIB kan kännas igen genom sambandet mellan tid och position. Emil jämför också passningar till BIB- och kollisionsprofiler. Det är en befintlig fysikmetod, inte ett neuralt nätverk.
- `LUP_Presentation_251002.pdf`, s. 21-22: en BIB-jet kan få en sen sammanfattad tid. Informationen inne i jeten kan hjälpa oss att förstå varifrån signalen kommer.

PDF:erna ligger lokalt i `reference/` och ingår inte i Git. Den offentliga översikten finns på [arXiv](https://arxiv.org/abs/2603.12306).

## Vilka modeller?

| Modell | Enkel beskrivning | Varför jag vill testa den |
|---|---|---|
| **XGBoost** | Beslutsträd på en rad variabler per jet. | Vår jämförelsemodell. Vi kan också lägga till kontrollerade geometrivariabler. |
| **Deep Sets / PFN** | Samma lilla nätverk läser varje cell. Resultaten summeras till en jetbeskrivning. | Ett enkelt första sätt att använda fler celluppgifter. Cellernas ordning ska inte ändra svaret. |
| **ParticleNet-Lite / GNN** | Celler blir punkter som kopplas till grannar. Nätverket lär samband mellan dem. | Kan undersöka hur position och tid hänger ihop lokalt. Lite är en mindre variant av ParticleNet. |
| **Particle Transformer / ParT** | Attention låter modellen väga samband mellan olika beståndsdelar. | Kan undersöka mer övergripande samband i jeten. En liten variant är rimlig att börja med. |

De tre sista är **neurala nätverk, alltså ANN**. De skiljer sig i hur informationen organiseras.
Vår anpassning blir att använda kalorimeterinformation och timing; originalmodellerna är inte färdiga BIB-taggare.

## Vad ska modellen få?

Vår nuvarande XGBoost får fem sammanfattningar. Ett cellnätverk skulle i stället kunna få en uppsättning vektorer:

$$
\mathbf{u}_i=(E_i,t_i,x_i,y_i,z_i,\text{detektorlager}_i,\ldots).
$$

Varje vektor beskriver en cell i **just den jet vi klassificerar**. Tidsosäkerhet och kvalitet kan läggas till efter kontroll av definitioner och ogiltiga värden.
Behåll även information om läget relativt strålen, till exempel absolut $z$ och avståndet $R$. Enbart koordinater relativt jetaxeln kan tappa den informationen.

Deep Sets kan skrivas enkelt som

$$
s_{\mathrm{BIB}}=\sigma\!\left[\rho\!\left(\sum_{i=1}^{N}\phi(\mathbf{u}_i)\right)\right].
$$

$\phi$ är samma lilla nätverk för varje cell. Summan samlar informationen och gör ordningen oviktig. $\rho$ och sigmoidfunktionen $\sigma$ ger jetens score. Scoren behöver valideras; den är inte automatiskt en fysisk sannolikhet.

## Praktisk ordning

1. Bekräfta med Emil hur celler och kluster kopplas till jets i just våra filer. Antal som stämmer räcker inte som bevis för rätt ordning.
2. Bygg en gemensam inläsning, maskning och skalning. Behåll lagrade celltider och standardtiden från ATLAS oförändrade.
3. Testa Deep Sets/PFN först, sedan ParticleNet-Lite och en liten ParT med samma urval.
4. Jämför med XGBoost och den befintliga BIB-passningen. Kontrollera särskilt försenade jets, LLP-bevarande och skillnader mellan MC och riktig data.
5. Använd en ny oberoende testmängd för att bekräfta kommande förbättringar. Det gamla testet har redan använts för undersökningar.

CERN:s GPU-miljö är en möjlig plats för träningen. Vi behöver först bekräfta åtkomst och välja SWAN eller ett batchjobb tillsammans med handledaren.
Full attention arbetar med cellpar och växer ungefär som $N^2$. Därför behöver vi mäta antalet celler per jet innan vi väljer modellstorlek, klusternivå eller eventuell begränsning av antalet indata.

## Frågor att ta med

- Ska nästa modell använda celler eller kluster? Vilken koppling till jets är garanterad i våra filer?
- Vilken GPU-miljö ska jag använda, och vilka resurser har jag tillgång till?
- Kan vi få fler försenade BIB-exempel och en separat körning eller LLP-signalpunkt för test?

## Läsning i den ordning jag skulle börja

1. [Energy Flow Networks: Deep Sets for Particle Jets](https://arxiv.org/abs/1810.05165) - PFN och idén med en oordnad samling. EFN har särskilda fysikrestriktioner; ett godtyckligt timingnätverk är inte automatiskt en EFN.
2. [ParticleNet](https://arxiv.org/abs/1902.08570) - EdgeConv och den mindre Lite-varianten, avsnitt III.
3. [Particle Transformer](https://arxiv.org/abs/2202.03772) - attention med information om par av beståndsdelar, avsnitt 4.
4. [ATLAS DeParT](https://cds.cern.ch/record/2878932/) - ett relevant ATLAS-exempel efter ParT. Den offentliga översikten beskriver det på s. 2-3; fulltexten i CDS kunde inte öppnas här.

GN2 använder främst spårinformation och LundNet använder jetens klustringshistoria. Jag skulle läsa dem senare, när vi vet att motsvarande indata finns för vårt problem.
