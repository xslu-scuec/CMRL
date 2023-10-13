/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCombinationMultiInputImageToImageMetric.h,v $
  Language:  C++
  Date:      $Date: 2013/09/29 12:23:34 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkCombinationMultiInputImageToImageMetric_h
#define __itkCombinationMultiInputImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkAdvancedTransform.h"
#include "itkMultiInputImageToImageMetricBase.h"
#include "itkSingleValuedPointSetToPointSetMetric.h"

namespace itk
{

/** \class CombinationMultiInputImageToImageMetric
 * \brief Combines multiinput metric with other metrics.
 *
 * This metric is meant to be used in the
 * MultiMetricMultiResolutionImageRegistrationMethodWithFeatures.
 *
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage>
class CombinationMultiInputImageToImageMetric :
  public MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
{
public:
  /** Standard class typedefs. */
  typedef CombinationMultiInputImageToImageMetric   Self;
  typedef MultiInputImageToImageMetricBase<
    TFixedImage, TMovingImage >                     Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( CombinationMultiInputImageToImageMetric, MultiInputImageToImageMetricBase );

  /** Define the New() method */
  itkNewMacro( Self );

  /** Constants for the image dimensions */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    TMovingImage::ImageDimension );
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    TFixedImage::ImageDimension );

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;
  typedef typename Superclass::AdvancedTransformType        TransformType;
  typedef typename TransformType::Pointer                   TransformPointer;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef typename Superclass::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass::RealType                     RealType;
  typedef typename Superclass::GradientPixelType            GradientPixelType;
  typedef typename Superclass::GradientImageType            GradientImageType;
  typedef typename Superclass::GradientImagePointer         GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType      GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer   GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType           FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer        FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType          MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer       MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename Superclass::ParametersType               ParametersType;

  /** Typedefs for the metrics. */
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >                             ImageMetricType;
  typedef typename ImageMetricType::Pointer                 ImageMetricPointer;
  typedef Superclass                                        MultiInputMetricType;
  typedef typename MultiInputMetricType::Pointer            MultiInputMetricPointer;
  typedef SingleValuedCostFunction                          SingleValuedCostFunctionType;
  typedef typename SingleValuedCostFunctionType::Pointer    SingleValuedCostFunctionPointer;

  typedef typename FixedImageType::PixelType                FixedImagePixelType;
  typedef typename MovingImageType::RegionType              MovingImageRegionType;
  typedef FixedArray< double,
    itkGetStaticConstMacro(MovingImageDimension) >          MovingImageDerivativeScalesType;

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

  /**
   * Get and set the metrics and their weights.
   **/

  /** Set the number of metrics to combine. */
  void SetNumberOfMetrics( unsigned int count );

  /** Get the number of metrics to combine. */
  itkGetConstMacro( NumberOfMetrics, unsigned int );

  /** Set metric i. It may be a SingleValuedCostFunction, instead of
   * a ImageToImageMetric, but the first one should be an
   * MultiInputImageToImageMetric in all cases.
   */
  void SetMetric( SingleValuedCostFunctionType * metric, unsigned int pos );

  /** Get metric i. */
  SingleValuedCostFunctionType * GetMetric( unsigned int count ) const;

  /** Set the weight for metric i. */
  void SetMetricWeight( double weight, unsigned int pos );

  /** Get the weight for metric i. */
  double GetMetricWeight( unsigned int pos ) const;

  /** Get the last computed value for metric i. */
  MeasureType GetMetricValue( unsigned int pos ) const;

  /** Get the last computed derivative for metric i. */
  const DerivativeType & GetMetricDerivative( unsigned int pos ) const;

  /** Get the last computed derivative magnitude for metric i. */
  double GetMetricDerivativeMagnitude( unsigned int pos ) const;

  /** Get the last computation time for metric i. */
  std::size_t GetMetricComputationTime( unsigned int pos ) const;
  
  /**
   * Set/Get functions for the metric components
   */

  /** Pass the transform to all sub metrics.  */
  virtual void SetTransform( TransformType * _arg );

  /** Pass an interpolator to a specific metric */
  virtual void SetInterpolator( InterpolatorType * _arg, unsigned int pos );

  /** Pass a fixed image to a specific metric */
  virtual void SetFixedImage( const FixedImageType *_arg, unsigned int pos );

  /** Pass a fixed image mask to a specific metric */
  virtual void SetFixedImageMask( FixedImageMaskType *_arg, unsigned int pos );

  /** Pass a fixed image region to a specific metric. */
  virtual void SetFixedImageRegion( const FixedImageRegionType _arg, unsigned int pos );

  /** Pass a moving image to a specific metric */
  virtual void SetMovingImage( const MovingImageType *_arg, unsigned int pos );

  /** Pass a moving image mask to a specific metric */
  virtual void SetMovingImageMask( MovingImageMaskType *_arg, unsigned int pos );
  
  /** Set the fixed image interpolators. */
  virtual void SetFixedImageInterpolator( FixedImageInterpolatorType *_arg, unsigned int pos );

  /** Get the number of pixels considered in the computation. Return the sum
   * of pixels counted by all metrics.
   */
  virtual const SizeValueType & GetNumberOfPixelsCounted( void ) const;

  /** Pass initialization to all sub metrics. */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /**
   * Combine all sub metrics by adding them.
   */

  /** The GetValue()-method. */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** The GetDerivative()-method. */
  virtual void GetDerivative(
    const ParametersType & parameters,
    DerivativeType & derivative ) const;

  /** The GetValueAndDerivative()-method. */
  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  /** Method to return the latest modified time of this object or any of its
   * cached ivars.
   */
  virtual unsigned long GetMTime() const;

protected:
  CombinationMultiInputImageToImageMetric();
  virtual ~CombinationMultiInputImageToImageMetric() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Store the metrics and the corresponding weights. */
  unsigned int                                      m_NumberOfMetrics;
  std::vector< SingleValuedCostFunctionPointer >    m_Metrics;
  std::vector< double >                             m_MetricWeights;
  mutable std::vector< MeasureType >                m_MetricValues;
  mutable std::vector< DerivativeType >             m_MetricDerivatives;
  mutable std::vector< double >                     m_MetricDerivativesMagnitude;
  mutable std::vector< std::size_t >                m_MetricComputationTime;

  /** Dummy image region and derivatives. */
  FixedImageRegionType        m_NullFixedImageRegion;
  DerivativeType              m_NullDerivative;

private:
  CombinationMultiInputImageToImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class CombinationMultiInputImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCombinationMultiInputImageToImageMetric.txx"
#endif

#endif // end #ifndef __itkCombinationMultiInputImageToImageMetric_h

