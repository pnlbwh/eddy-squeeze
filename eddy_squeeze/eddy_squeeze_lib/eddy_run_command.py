class EddyRunCommand():
    def __init__(self):
        pass


def eddy(echo_spacing, img_in, bvec, bval, mask,
         eddy_out_prefix, repol_on=True):
    '''
    Run FSL eddy

    Parameters:
    echo_spacing (float) : echo spacing of the diffusion data
    other inputs (string)

    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;92bd6f89.1403
        The echo spacing in the Siemens PDF does need to be divided
        by the ipat/GRAPPA factor.  Multi-band has no effect on
        echo spacing.

    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;92136ade.1506
        What I can say is that in _most_ cases it doesn’t really matter
        what you put in the last column. The value is essentially used
        internally to calculate "observed_distortions->estimated_field
        ->estimated_distortions” where the value is used at both -> .
        That means that if you get it wrong, the two errors will cancel.

        It is only really important if
        1. You want your estimated fields to be correctly scaled in Hz
        2. You have acquisitions with different readout times in your
           data set

        Otherwise you can typically just go with 0.05.
        And also make sure that you use the same values for topup and
        eddy if you are using them together.
        Jesper

    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq
        There are some special cases where it matters to get the --acqp
        file right, but unless you know exactly what you are doing it
        is generally best to avoid those cases.
        They would be
        - If you acquire data with PE direction along two different
          axes (i.e. the x- and y-axis). In that case you need to get
          the signs right for the columns indicating the PE. But you can
          always use trial and error to find the correct combination.
        - If you acquire data with different readout times. In that case
          you need at the very least to get the ratio between the times
          right.
        - If you use a non-topup derived fieldmap, such as for example
          a dual echo-time gradient echo fieldmap, that you feed into
          eddy as the --field parameter. In this case you need to get
          all signs and times right, both when creating the field (for
          example using prelude) and when specifying its use in eddy
          through the --acqp.

    '''
    eddy_out_dir = dirname(eddy_out_prefix)

    # index
    data_img = nb.load(img_in)
    index_array = np.tile(1, data_img.shape[-1])
    index_loc = join(eddy_out_dir, 'index.txt')
    np.savetxt(index_loc, index_array, fmt='%d', newline=' ')

    # acqp
    acqp_num = (128-1) * echo_spacing * 0.001
    acqp_line = '0 -1 0 {}'.format(acqp_num)
    acqp_loc = join(eddy_out_dir, 'acqp.txt')
    with open(acqp_loc, 'w') as f:
        f.write(acqp_line)

    if repol_on:
        # eddy_command
        eddy_command = '/data/pnl/soft/pnlpipe3/fsl/bin/eddy_openmp \
            --imain={data} \
            --mask={mask} \
            --index={index} \
            --acqp={acqp} \
            --bvecs={bvecs} \
            --bvals={bvals} \
            --repol \
            --out={out}'.format(data=img_in,
                                mask=mask,
                                index=index_loc,
                                acqp=acqp_loc,
                                bvecs=bvec,
                                bvals=bval,
                                out=eddy_out_prefix)
    else:
        # eddy_command
        eddy_command = '/data/pnl/soft/pnlpipe3/fsl/bin/eddy_openmp \
            --imain={data} \
            --mask={mask} \
            --index={index} \
            --acqp={acqp} \
            --bvecs={bvecs} \
            --bvals={bvals} \
            --out={out}'.format(data=img_in,
                                mask=mask,
                                index=index_loc,
                                acqp=acqp_loc,
                                bvecs=bvec,
                                bvals=bval,
                                out=eddy_out_prefix)

    print(re.sub(r'\s+', ' ', eddy_command))
    run(eddy_command)


def eddy_qc(echo_spacing, bvec, bval, mask, eddy_out_prefix):
    '''
    Run FSL eddy qc

    Parameters:
        echo_spacing (float) : echo spacing of the diffusion data
        other inputs (string)

        bvec is not currently used
    '''
    eddy_out_dir = dirname(eddy_out_prefix)
    quad_outdir = '{}.qc'.format(eddy_out_prefix)
    index_loc = join(eddy_out_dir, 'index.txt')
    acqp_loc = join(eddy_out_dir, 'acqp.txt')

    command = f'{config["eddy_quad_location"] {eddy_out_prefix} \
            -idx {index_loc} \
            -par {acqp_loc} \
            -m {mask} \
            -b {bval} \
            -o {quad_outdir}'

    run(command)

def eddy_squad(qc_dirs, outdir):
    try:
        os.mkdir(outdir)
    except:
        pass
    out_text_file = join(outdir, 'eddy_quad_dirs.txt')
    with open(out_text_file, 'w') as f:
        for qc_dir in qc_dirs:
            f.write('{}\n'.format(qc_dir))

    command = f'{config["eddy_squad_location"] {out_text_file} -o {outdir}'
    run(command)
