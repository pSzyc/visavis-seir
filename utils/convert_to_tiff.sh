for fig in fig1 fig2 fig3 fig4 fig5 fig6 figS1 figS2 figS4 figS6 figS7 figS2T
do
    convert -compress LZW -alpha remove figures-ready/$fig.png figures-ready/tiffs/$fig.tiff
done