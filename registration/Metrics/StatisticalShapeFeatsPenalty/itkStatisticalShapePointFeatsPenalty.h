/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkStatisticalShapePointFeatsPenalty_h
#define __itkStatisticalShapePointFeatsPenalty_h

/** Includes for the Superclass. */
#include "itkMultiInputImageToImageMetricBase.h"

/** Includes for the PointSet. */
#include "itkSingleValuedPointSetToPointSetMetric.h"

/** Include for the spatial derivatives. */
#include "itkArray2D.h"

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_math.h>
#include <vnl/vnl_vector.h>
#include <vnl/algo/vnl_real_eigensystem.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <vnl/algo/vnl_svd_economy.h>


namespace itk
{
/**
 * \class StatisticalShapePointFeatsPenalty
 *
 * \brief Computes the Mahalanobis distance between the transformed 
 *  shape features and a mean shape features. A model mean and 
 *  covariance are required.
 *
 * Note that the feature image are given beforehand, and that values
 * are calculated by interpolation on the transformed point. For some
 * features, it would be better (but slower) to first apply the transform
 * on the image and then recalculate the feature.
 *
 * \ingroup RegistrationMetrics
 */

template <class TFixedImage, class TMovingImage>
class StatisticalShapePointFeatsPenalty :
  public MultiInputImageToImageMetricBase<TFixedImage, TMovingImage>
{
public:

  /** Standard itk. */
  typedef StatisticalShapePointFeatsPenalty                 Self;
  typedef MultiInputImageToImageMetricBase<
    TFixedImage, TMovingImage >                             Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StatisticalShapePointFeatsPenalty,
    MultiInputImageToImageMetricBase );

  /** Typedefs from the superclass. */
  typedef typename
    Superclass::CoordinateRepresentationType              CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
  typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** Typedef's for storing multiple inputs. */
  typedef typename Superclass::FixedImageVectorType       FixedImageVectorType;
  typedef typename Superclass::FixedImageMaskVectorType   FixedImageMaskVectorType;
  typedef typename Superclass::FixedImageRegionVectorType FixedImageRegionVectorType;
  typedef typename Superclass::MovingImageVectorType      MovingImageVectorType;
  typedef typename Superclass::MovingImageMaskVectorType  MovingImageMaskVectorType;
  typedef typename Superclass::InterpolatorVectorType     InterpolatorVectorType;
  typedef typename Superclass::FixedImageInterpolatorVectorType FixedImageInterpolatorVectorType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

  /** Typedef for the PointSetMetric. */
  typedef PointSet< CoordinateRepresentationType,
    TFixedImage::ImageDimension,
    DefaultStaticMeshTraits<
      CoordinateRepresentationType,
      TFixedImage::ImageDimension,
      TFixedImage::ImageDimension,
      CoordinateRepresentationType, CoordinateRepresentationType,
      CoordinateRepresentationType > >                      FixedPointSetType;
  typedef PointSet< CoordinateRepresentationType,
    TMovingImage::ImageDimension,
    DefaultStaticMeshTraits<
      CoordinateRepresentationType,
      TMovingImage::ImageDimension,
      TMovingImage::ImageDimension,
      CoordinateRepresentationType, CoordinateRepresentationType,
      CoordinateRepresentationType > >                      MovingPointSetType;
  typedef SingleValuedPointSetToPointSetMetric<
    FixedPointSetType, MovingPointSetType >                 PointSetMetricType;
  typedef typename PointSetMetricType
	::FixedPointSetConstPointer                             FixedPointSetConstPointer;
  typedef typename PointSetMetricType
	::MovingPointSetConstPointer                            MovingPointSetConstPointer;

  typedef typename PointSetMetricType::PointIterator        PointIterator;
  typedef typename PointSetMetricType::PointDataIterator    PointDataIterator;

  typedef typename PointSetMetricType::InputPointType       InputPointType;
  typedef typename PointSetMetricType::OutputPointType      OutputPointType;

  /** Typedef for the member variables. */
  typedef vnl_vector< double >                           VnlVectorType;
  typedef vnl_matrix< double >                           VnlMatrixType;
  typedef typename std::vector< VnlVectorType * >        ProposalDerivativeType;

  /** Typedefs for multi-threading. */
  typedef typename Superclass::ThreaderType                SSPThreaderType;
  typedef typename Superclass::ThreadInfoType              SSPThreadInfoType;

  /**
   * *** Standard metric stuff: ***
   */

  /** Initialize the metric. */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & Derivative ) const;

  /** Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;

  /** Set the fixed pointset.  */
  itkSetConstObjectMacro( FixedPointSet, FixedPointSetType );

  /** Get the fixed pointset.  */
  itkGetConstObjectMacro( FixedPointSet, FixedPointSetType );

  /** Set the mean vector.  */
  itkSetConstObjectMacro( MeanVector, VnlVectorType );

  /** Set the covariance matrix.  */
  itkSetConstObjectMacro( CovarianceMatrix, VnlMatrixType );

  /** Set the ShrinkageIntensityFeats parameter.  */
  itkSetClampMacro( ShrinkageIntensityFeats, MeasureType, 0.0, 1.0 );

  /** Set the BaseVarianceFeats parameter.  */
  itkSetClampMacro( BaseVarianceFeats, MeasureType,
    -1.0, NumericTraits< MeasureType >::max() );

protected:

  /** Constructor. */
  StatisticalShapePointFeatsPenalty();

  /** Destructor. */
  virtual ~StatisticalShapePointFeatsPenalty();

  /** PrintSelf. */
  virtual void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Helper structs that multi-threads the computation of
  * the metric derivative using ITK threads.
  */
  struct SSPMultiThreaderParameterType
  {
	  // To give the threads access to all members.
	  StatisticalShapePointFeatsPenalty * st_Metric;
	  // Used for accumulating derivatives
	  DerivativeValueType * st_DerivativePointer;
  };
  mutable SSPMultiThreaderParameterType m_SSPThreaderMetricParameters;

  /** Most metrics will perform multi-threading by letting
  * each thread compute a part of the value and derivative.
  *
  * These parameters are initialized at every call of GetValueAndDerivative
  * in the function InitializeThreadingParameters(). Since GetValueAndDerivative
  * is const, also InitializeThreadingParameters should be const, and therefore
  * these member variables are mutable.
  */

  // test per thread struct with padding and alignment
  struct SSPGetValueAndDerivativePerThreadStruct
  {
	  VnlVectorType      st_differenceVector;
	  MeasureType        st_value;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT, SSPGetValueAndDerivativePerThreadStruct,
	  PaddedSSPGetValueAndDerivativePerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT, PaddedSSPGetValueAndDerivativePerThreadStruct,
	  AlignedSSPGetValueAndDerivativePerThreadStruct);
  mutable AlignedSSPGetValueAndDerivativePerThreadStruct * m_SSPGetValueAndDerivativePerThreadVariables;
  mutable ThreadIdType                                     m_SSPGetValueAndDerivativePerThreadVariablesSize;

  /** Initialize some multi-threading related parameters. */
  virtual void InitializeSSPThreadingParameters(void) const;

  /** SSPCalculateDerivative threader callback function. */
  static ITK_THREAD_RETURN_TYPE SSPCalculateDerivativeThreaderCallback(void * arg);

  /** After calculating the derivatives for multi-threading. */
  void AfterSSPThreadedCalculateDerivative(void) const;

private:
  StatisticalShapePointFeatsPenalty(const Self&);                 //purposely not implemented
  void operator=(const Self&);                                    //purposely not implemented

  /** Typedef's for the computation of the derivative. */
  typedef typename Superclass::FixedImagePointType              FixedImagePointType;
  typedef typename Superclass::MovingImagePointType             MovingImagePointType;
  typedef typename Superclass::MovingImageDerivativeType        MovingImageDerivativeType;
  typedef typename Superclass::MovingImageContinuousIndexType   MovingImageContinuousIndexType;
  typedef typename DerivativeType::ValueType                    DerivativeValueType;
  typedef typename TransformJacobianType::ValueType             TransformJacobianValueType;
  typedef std::vector<TransformJacobianType>                    TransformJacobianContainerType;
  typedef std::vector<NonZeroJacobianIndicesType>               TransformJacobianIndicesContainerType;
  typedef Array2D<double>                                       SpatialDerivativeType;
  typedef std::vector<SpatialDerivativeType>                    SpatialDerivativeContainerType;

  /** This function copies the point features in proposal vector
   * and the point derivatives in proposal derivative vector.
   */
  void FillProposalVector(const OutputPointType & fixedPoint,
	const unsigned int vertexindex) const;

  void FillProposalDerivative(const OutputPointType & fixedPoint,
	const unsigned int vertexindex) const; 

  void CalculateValue( MeasureType & value, VnlVectorType & differenceVector ) const;

  void CalculateDerivative( DerivativeType & derivative, const MeasureType & value,
    const VnlVectorType & differenceVector ) const;
  
  /** This function calculates the spatial derivative of the
   * featureNr feature image at the point mappedPoint.
   * \todo move this to base class.
   */
  
  
  /** Member variables. */
  FixedPointSetConstPointer   m_FixedPointSet;
  
  const VnlVectorType *       m_MeanVector;
  const VnlMatrixType *       m_CovarianceMatrix;

  double                      m_ShrinkageIntensityFeats;
  double                      m_BaseVarianceFeats;

  unsigned int                     m_ProposalLength;
  mutable VnlVectorType            m_ProposalVector;
  mutable ProposalDerivativeType * m_ProposalDerivative;
  VnlMatrixType *                  m_InverseCovarianceMatrix;

 }; // end class StatisticalShapePointFeatsPenalty

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStatisticalShapePointFeatsPenalty.txx"
#endif

#endif // end #ifndef __itkStatisticalShapePointFeatsPenalty_h

