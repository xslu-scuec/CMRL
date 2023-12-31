/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMultiMetricMultiResolutionImageRegistrationMethod
                       WithFeatures.h,v $
  Language:  C++
  Date:      $Date: 2013/09/29 12:23:34 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkMultiMetricMultiResolutionImageRegistrationMethodWithFeatures_h
#define __itkMultiMetricMultiResolutionImageRegistrationMethodWithFeatures_h

#include "itkMultiResolutionImageRegistrationMethod2.h"
#include "itkCombinationMultiInputImageToImageMetric.h"
#include <vector>

/** defines a method that calls the same method
 * with an extra 0 argument.
 */
#define itkSimpleSetMacro(_name,_type) \
  virtual void Set##_name( _type _arg ) \
  { \
    this->Set##_name ( _arg, 0 ); \
  }

/** defines for example: SetNumberOfInterpolators(). */
#define itkSetNumberOfMacro(_name) \
  virtual void SetNumberOf##_name##s(unsigned int _arg) \
  { \
    if ( this->m_##_name##s.size() != _arg ) \
    { \
      this->m_##_name##s.resize( _arg ); \
      this->Modified(); \
    } \
  }

/** defines for example: GetNumberOfInterpolators() */
#define itkGetNumberOfMacro(_name) \
  virtual unsigned int GetNumberOf##_name##s(void) const \
  { \
    return this->m_##_name##s.size(); \
  }


namespace itk
{

/** \class MultiMetricMultiResolutionImageRegistrationMethodWithFeatures
 * \brief Base class for multi-resolution image registration methods
 *
 * This class is an extension of the itk class
 * MultiResolutionImageRegistrationMethod. It allows the use
 * of multiple metrics, multiple images,
 * multiple interpolators, and/or multiple image pyramids.
 *
 * You may also set an interpolator/fixedimage/etc to NULL, if you
 * happen to know that the corresponding metric is not an
 * ImageToImageMetric, but a regularizer for example (which does
 * not need an image.
 *
 *
 * \sa ImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod2
 * \ingroup RegistrationFilters
 */

template <typename TFixedImage, typename TMovingImage>
class ITK_EXPORT MultiMetricMultiResolutionImageRegistrationMethodWithFeatures :
  public MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiMetricMultiResolutionImageRegistrationMethodWithFeatures  Self;
  typedef MultiResolutionImageRegistrationMethod2<
    TFixedImage, TMovingImage>                                           Superclass;
  typedef SmartPointer<Self>                                             Pointer;
  typedef SmartPointer<const Self>                                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultiMetricMultiResolutionImageRegistrationMethodWithFeatures,
    MultiResolutionImageRegistrationMethod2 );

  /**  Superclass types */
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;
  typedef typename Superclass::FixedImageRegionPyramidType  FixedImageRegionPyramidType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;

  typedef typename Superclass::MetricType                   MetricType;
  typedef typename Superclass::MetricPointer                MetricPointer;
  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef typename Superclass::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass::OptimizerType                OptimizerType;
  typedef typename OptimizerType::Pointer                   OptimizerPointer;
  typedef typename Superclass::FixedImagePyramidType        FixedImagePyramidType;
  typedef typename Superclass::FixedImagePyramidPointer     FixedImagePyramidPointer;
  typedef typename Superclass::MovingImagePyramidType       MovingImagePyramidType;
  typedef typename
    Superclass::MovingImagePyramidPointer                   MovingImagePyramidPointer;

  typedef typename Superclass::TransformOutputType          TransformOutputType;
  typedef typename Superclass::TransformOutputPointer       TransformOutputPointer;
  typedef typename
    Superclass::TransformOutputConstPointer                 TransformOutputConstPointer;

  typedef typename Superclass::ParametersType               ParametersType;
  typedef typename Superclass::DataObjectPointer            DataObjectPointer;

  typedef std::vector< FixedImageRegionPyramidType >        FixedImageRegionPyramidVectorType;

  /** Extra typedefs */
  typedef CombinationMultiInputImageToImageMetric<
    FixedImageType, MovingImageType >                       CombinationMultiInputMetricType;
  typedef typename 
    CombinationMultiInputMetricType::Pointer                CombinationMultiInputMetricPointer;
  typedef typename CombinationMultiInputMetricType
    ::FixedImageVectorType                                  FixedImageVectorType;
  typedef typename CombinationMultiInputMetricType
    ::FixedImageRegionVectorType                            FixedImageRegionVectorType;
  typedef typename CombinationMultiInputMetricType
    ::MovingImageVectorType                                 MovingImageVectorType;
  typedef typename CombinationMultiInputMetricType
    ::InterpolatorVectorType                                InterpolatorVectorType;
  typedef typename CombinationMultiInputMetricType
    ::FixedImageInterpolatorType                            FixedImageInterpolatorType;
  typedef typename CombinationMultiInputMetricType
    ::FixedImageInterpolatorVectorType                      FixedImageInterpolatorVectorType;
  typedef std::vector<FixedImagePyramidPointer>             FixedImagePyramidVectorType;
  typedef std::vector<MovingImagePyramidPointer>            MovingImagePyramidVectorType;
  
  /** The following methods all have a similar pattern. The
   * SetFixedImage() just calls SetFixedImage(0).
   * SetFixedImage(0) also calls the Superclass::SetFixedImage(). This
   * is defined by the itkSimpleSetMacro.
   * GetFixedImage() just returns GetFixedImage(0) == Superclass::m_FixedImage.
   */

  /** Set/Get the Fixed image. */
  virtual void SetFixedImage( const FixedImageType * _arg, unsigned int pos );
  virtual const FixedImageType * GetFixedImage( unsigned int pos ) const;
  virtual const FixedImageType * GetFixedImage( void ) const
  { return this->GetFixedImage( 0 ); }
  itkSimpleSetMacro( FixedImage, const FixedImageType * );
  itkSetNumberOfMacro( FixedImage );
  itkGetNumberOfMacro( FixedImage );

  /** Set/Get the Fixed image region. */
  virtual void SetFixedImageRegion( FixedImageRegionType _arg, unsigned int pos );
  virtual const FixedImageRegionType & GetFixedImageRegion( unsigned int pos ) const;
  virtual const FixedImageRegionType & GetFixedImageRegion( void ) const
  { return this->GetFixedImageRegion( 0 ); }
  itkSimpleSetMacro( FixedImageRegion, const FixedImageRegionType );
  itkSetNumberOfMacro( FixedImageRegion );
  itkGetNumberOfMacro( FixedImageRegion );

  /** Set/Get the FixedImagePyramid. */
  virtual void SetFixedImagePyramid( FixedImagePyramidType * _arg, unsigned int pos );
  virtual FixedImagePyramidType * GetFixedImagePyramid( unsigned int pos ) const;
  virtual FixedImagePyramidType * GetFixedImagePyramid( void )
  { return this->GetFixedImagePyramid( 0 ); }
  itkSimpleSetMacro( FixedImagePyramid, FixedImagePyramidType * );
  itkSetNumberOfMacro( FixedImagePyramid );
  itkGetNumberOfMacro( FixedImagePyramid );

  /** Set/Get the Moving image. */
  virtual void SetMovingImage( const MovingImageType * _arg, unsigned int pos );
  virtual const MovingImageType * GetMovingImage( unsigned int pos ) const;
  virtual const MovingImageType * GetMovingImage( void ) const
  { return this->GetMovingImage( 0 ); }
  itkSimpleSetMacro( MovingImage, const MovingImageType * );
  itkSetNumberOfMacro( MovingImage );
  itkGetNumberOfMacro( MovingImage );

  /** Set/Get the MovingImagePyramid. */
  virtual void SetMovingImagePyramid( MovingImagePyramidType * _arg, unsigned int pos );
  virtual MovingImagePyramidType * GetMovingImagePyramid( unsigned int pos ) const;
  virtual MovingImagePyramidType * GetMovingImagePyramid( void )
  { return this->GetMovingImagePyramid( 0 ); }
  itkSimpleSetMacro( MovingImagePyramid, MovingImagePyramidType * );
  itkSetNumberOfMacro( MovingImagePyramid );
  itkGetNumberOfMacro( MovingImagePyramid );

  /** Set/Get the Interpolator. */
  virtual void SetInterpolator( InterpolatorType * _arg, unsigned int pos );
  virtual InterpolatorType * GetInterpolator( unsigned int pos ) const;
  virtual InterpolatorType * GetInterpolator( void )
  { return this->GetInterpolator( 0 ); }
  itkSimpleSetMacro( Interpolator, InterpolatorType * );
  itkSetNumberOfMacro( Interpolator );
  itkGetNumberOfMacro( Interpolator );

  /** Set/Get the FixedImageInterpolator. */
  virtual void SetFixedImageInterpolator( FixedImageInterpolatorType * _arg, unsigned int pos );
  virtual FixedImageInterpolatorType * GetFixedImageInterpolator( unsigned int pos ) const;
  virtual FixedImageInterpolatorType * GetFixedImageInterpolator( void )
  { return this->GetFixedImageInterpolator( 0 ); }
  itkSimpleSetMacro( FixedImageInterpolator, FixedImageInterpolatorType * );
  itkSetNumberOfMacro( FixedImageInterpolator );
  itkGetNumberOfMacro( FixedImageInterpolator );

  /** Set the Metric. Reimplement this method to check if
   * the metric is a combination multiinput metric.
   * GetMetric returns the combination multiinput metric.
   * By default, a combination multiinput metric is already set 
   * on constructing this class.
   */
  virtual void SetMetric( MetricType * _arg );

  /** Get the metric as a pointer to a combination multiinput metric type.
   * Use this method to setup the combination multiinput metric (set weights,
   * nrofmetrics, submetrics, etc.
   */
  virtual CombinationMultiInputMetricType * GetCombinationMultiInputMetric( void ) const
  {
    return this->m_CombinationMultiInputMetric.GetPointer();
  }

  /** Method to return the latest modified time of this object or
   * any of its cached ivars.
   */
  unsigned long GetMTime( void ) const;

  /** Get the last transformation parameters visited by
   * the optimizer. Return the member variable declared in this class,
   * and not that of the superclass (which is declared private).
   */
  virtual const ParametersType & GetLastTransformParameters( void ) const
  {
    return this->m_LastTransformParameters;
  }

protected:

  /** Constructor. */
  MultiMetricMultiResolutionImageRegistrationMethodWithFeatures();

  /** Destructor. */
  virtual ~MultiMetricMultiResolutionImageRegistrationMethodWithFeatures() {};

  /** PrintSelf. */
  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the registration.
   */
  virtual void GenerateData();

  /** Initialize by setting the interconnects between the components.
   * This method is executed at every level of the pyramid with the
   * values corresponding to this resolution .
   */
  virtual void Initialize() throw (ExceptionObject);

  /** Compute the size of the fixed region for each level of the pyramid. */
  virtual void PreparePyramids( void );

  /** Function called by PreparePyramids, which checks if the user input
   * regarding the image pyramids is ok.
   */
  virtual void CheckPyramids( void ) throw (ExceptionObject);

  /** Function called by Initialize, which checks if the user input is ok. */
  virtual void CheckOnInitialize( void ) throw (ExceptionObject);

  /** Containers for the pointers supplied by the user */
  FixedImageVectorType               m_FixedImages;
  MovingImageVectorType              m_MovingImages;
  FixedImageRegionVectorType         m_FixedImageRegions;
  FixedImagePyramidVectorType        m_FixedImagePyramids;
  MovingImagePyramidVectorType       m_MovingImagePyramids;
  InterpolatorVectorType             m_Interpolators;
  FixedImageInterpolatorVectorType   m_FixedImageInterpolators;

  /** This vector is filled by the PreparePyramids function. */
  FixedImageRegionPyramidVectorType  m_FixedImageRegionPyramids;

  /** Dummy image region */
  FixedImageRegionType               m_NullFixedImageRegion;

  /** Variables already defined in the superclass, but as private...  */
  ParametersType                     m_LastTransformParameters;

  /** A shortcut to m_Metric of type CombinationMultiInputMetricPointer. */
  CombinationMultiInputMetricPointer m_CombinationMultiInputMetric;

private:
  MultiMetricMultiResolutionImageRegistrationMethodWithFeatures(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class MultiMetricMultiResolutionImageRegistrationMethodWithFeatures


} // end namespace itk

#undef itkSetNumberOfMacro
#undef itkGetNumberOfMacro
#undef itkSimpleSetMacro

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiMetricMultiResolutionImageRegistrationMethodWithFeatures.txx"
#endif

#endif // end #ifndef __itkMultiMetricMultiResolutionImageRegistrationMethodWithFeatures_h

