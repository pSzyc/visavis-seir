for fig in fig1 fig2 fig3 fig4 fig5 fig6 fig7 figS1 figS2 figS3 figS4 figS5 figS6 figS7 figS8 figS9
do
    convert -compress LZW -alpha remove figures-ready/$fig.png figures-ready/tiffs/$fig.tiff
done