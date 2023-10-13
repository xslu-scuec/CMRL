/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxStatisticalShapeFeatsPenalty_H__
#define __elxStatisticalShapeFeatsPenalty_H__

#include "elxIncludes.h"
#include "itkStatisticalShapePointFeatsPenalty.h"

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vcl_iostream.h>

namespace elastix
{
using namespace itk;

  /**
   * \class StatisticalShapeFeatsPenalty
   * \brief A metric based on the
   * itk::StatisticalShapePointFeatsPenalty.
   *
   * The parameters used in this class are:
   * \parameter Metric: Select this metric as follows:\n
   *    <tt>(Metric "StatisticalShapeFeatsPenalty")</tt>
   * \parameter ShrinkageIntensityFeats: The mixing ratio ($\beta$) of the provided
   *    covariance matrix and an identity matrix.
   *    $\Sigma' = (1-\beta)\Sigma + \beta \sigma_0^2 I$
   *    Can be defined for each resolution\n
   *    example: <tt>(ShrinkageIntensityFeats 0.2)</tt>
   *    Choose a value between 0.0 and 1.0. The default is 0.2.
   * \parameter BaseVarianceFeats: The width ($\sigma_0^2$) of the non-informative prior.
   *    Can be defined for each resolution\n
   *    example: <tt>(BaseVarianceFeats 10.0)</tt>
   *    The default is 10.0 for all resolutions.
   *
   * \sa StatisticalShapePenalty
   * \ingroup Metrics
   */

  template <class TElastix>
  class StatisticalShapeFeatsPenalty :
    public
      itk::StatisticalShapePointFeatsPenalty<
        typename MetricBase<TElastix>::FixedImageType,
        typename MetricBase<TElastix>::MovingImageType >,
    public MetricBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef StatisticalShapeFeatsPenalty                  Self;
    typedef itk::StatisticalShapePointFeatsPenalty<
      typename MetricBase<TElastix>::FixedImageType,
      typename MetricBase<TElastix>::MovingImageType >    Superclass1;
    typedef MetricBase<TElastix>                          Superclass2;
    typedef itk::SmartPointer<Self>                       Pointer;
    typedef itk::SmartPointer<const Self>                 ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( StatisticalShapeFeatsPenalty,
      itk::StatisticalShapePointFeatsPenalty );

    /** Name of this class.
     * Use this name in the parameter file to select this specific metric. \n
     * example: <tt>(Metric "StatisticalShapeFeatsPenalty")</tt>\n
     */
    elxClassNameMacro( "StatisticalShapeFeatsPenalty" );

    /** Typedefs inherited from the superclass.*/
    typedef typename Superclass1::TransformType             TransformType;
    typedef typename Superclass1::TransformPointer          TransformPointer;
	typedef typename Superclass1::TransformParametersType   TransformParametersType;
    typedef typename Superclass1::TransformJacobianType     TransformJacobianType;
    typedef typename Superclass1::FixedImageType            FixedImageType;
    typedef typename Superclass1::MovingImageType           MovingImageType;
    typedef typename Superclass1::FixedImageConstPointer    FixedImageConstPointer;
    typedef typename Superclass1::MovingImageConstPointer   MovingImageConstPointer;
    typedef typename Superclass1::FixedImageMaskType        FixedImageMaskType;
    typedef typename Superclass1::FixedImageMaskPointer     FixedImageMaskPointer;
    typedef typename Superclass1::MovingImageMaskType       MovingImageMaskType;
    typedef typename Superclass1::MovingImageMaskPointer    MovingImageMaskPointer;
    typedef typename Superclass1::MeasureType               MeasureType;
    typedef typename Superclass1::DerivativeType            DerivativeType;
    typedef typename Superclass1::ParametersType            ParametersType;

	typedef typename Superclass1::FixedPointSetType            FixedPointSetType;
    typedef typename Superclass1::FixedPointSetConstPointer    FixedPointSetConstPointer;
    typedef typename Superclass1::MovingPointSetType           MovingPointSetType;
    typedef typename Superclass1::MovingPointSetConstPointer   MovingPointSetConstPointer;
    typedef typename Superclass1::InputPointType               InputPointType;
    typedef typename Superclass1::OutputPointType              OutputPointType;

	typedef typename OutputPointType::CoordRepType             CoordRepType;

    /** The fixed image dimension */
    itkStaticConstMacro( FixedImageDimension, unsigned int,
      FixedImageType::ImageDimension);
    /** The moving image dimension. */
    itkStaticConstMacro( MovingImageDimension, unsigned int,
      MovingImageType::ImageDimension );

    /** Typedef's inherited from Elastix. */
    typedef typename Superclass2::ElastixType               ElastixType;
    typedef typename Superclass2::ElastixPointer            ElastixPointer;
    typedef typename Superclass2::ConfigurationType         ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
    typedef typename Superclass2::RegistrationType          RegistrationType;
    typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
    typedef typename Superclass2::ITKBaseType               ITKBaseType;

    /** Typedefs for the image and point set. */
    typedef FixedImageType          ImageType;
	typedef FixedPointSetType       PointSetType;

    /** Execute stuff before the registration:
     * \li Read and set the fixed pointset.
     * \li Read and set the mean vector.
     * \li Read and set the covariance matrix.
     */
    virtual void BeforeRegistration( void );

    /** Execute stuff before each new pyramid resolution:
     * \li Get and set ShrinkageIntensityFeats.
     * \li Get and set BaseVarianceFeats.
     */
    virtual void BeforeEachResolution( void );

    /** Sets up a timer to measure the intialisation time and
     * calls the Superclass' implementation.
     */
    virtual void Initialize(void) throw (itk::ExceptionObject);

	/** Function to read the corresponding points. */
    unsigned int ReadShape(
      const std::string & ShapeFileName,
      typename PointSetType::Pointer & pointSet,
      const typename ImageType::ConstPointer image );

  protected:

    /** The constructor. */
    StatisticalShapeFeatsPenalty() {};
    /** The destructor. */
    virtual ~StatisticalShapeFeatsPenalty() {}

  private:

    /** The private constructor. */
    StatisticalShapeFeatsPenalty( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );                  // purposely not implemented

  }; // end class StatisticalShapeFeatsPenalty


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxStatisticalShapeFeatsPenalty.hxx"
#endif

#endif // end #ifndef __elxStatisticalShapeFeatsPenalty_H__
