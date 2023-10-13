/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCombinationMultiInputImageToImageMetric.txx,v $
  Language:  C++
  Date:      $Date: 2013/09/29 12:23:34 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef _itkCombinationMultiInputImageToImageMetric_txx
#define _itkCombinationMultiInputImageToImageMetric_txx

#include "itkCombinationMultiInputImageToImageMetric.h"
#include "itkMath.h"


/** Macros to reduce some copy-paste work.
 * These macros provide the implementation of
 * all Set/GetFixedImage, Set/GetInterpolator etc methods
 *
 * The macros are undef'ed at the end of this file
 */
 
/** For setting objects, implement two methods */
#define itkImplementationSetObjectMacro1(_name, _type1, _type2 ) \
template <class TFixedImage, class TMovingImage> \
void \
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage> \
::Set##_name ( _type1 _type2 *_arg, unsigned int pos ) \
{ \
  if ( pos == 0 ) \
  { \
    for ( unsigned int i = 1; i < this->GetNumberOfMetrics(); i++ ) \
    { \
      ImageMetricType    * testPtr1 \
        = dynamic_cast<ImageMetricType *>( this->GetMetric( i ) ); \
	  if ( testPtr1 ) \
      { \
        testPtr1->Set##_name ( _arg ); \
      } \
    } \
  } \
  MultiInputMetricType    * testPtr2 \
        = dynamic_cast<MultiInputMetricType *>( this->GetMetric( 0 ) ); \
  if ( testPtr2 ) \
  { \
    testPtr2->Set##_name ( _arg, pos ); \
  } \
}  // comments for allowing ; after calling the macro

#define itkImplementationSetObjectMacro2(_name, _type1, _type2 ) \
template <class TFixedImage, class TMovingImage> \
void \
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage> \
::Set##_name ( _type1 _type2 *_arg, unsigned int pos ) \
{ \
  if ( pos == 0 ) \
  { \
    for ( unsigned int i = 1; i < this->GetNumberOfMetrics(); i++ ) \
    { \
      ImageMetricType    * testPtr1 \
        = dynamic_cast<ImageMetricType *>( this->GetMetric( i ) ); \
	  PointSetMetricType * testPtr2 \
        = dynamic_cast<PointSetMetricType *>( this->GetMetric( i ) ); \
	  if ( testPtr1 ) \
      { \
        testPtr1->Set##_name ( _arg ); \
      } \
	  else if ( testPtr2 ) \
      { \
        testPtr2->Set##_name ( _arg ); \
      } \
    } \
  } \
  MultiInputMetricType    * testPtr3 \
        = dynamic_cast<MultiInputMetricType *>( this->GetMetric( 0 ) ); \
  if ( testPtr3 ) \
  { \
    testPtr3->Set##_name ( _arg, pos ); \
  } \
}  // comments for allowing ; after calling the macro


namespace itk
{

itkImplementationSetObjectMacro1( Interpolator, , InterpolatorType );
itkImplementationSetObjectMacro2( FixedImageMask, , FixedImageMaskType );
itkImplementationSetObjectMacro2( MovingImageMask, , MovingImageMaskType );
itkImplementationSetObjectMacro1( FixedImage, const, FixedImageType );
itkImplementationSetObjectMacro1( MovingImage, const, MovingImageType );


/**
 * ********************* Constructor ****************************
 */

template <class TFixedImage, class TMovingImage>
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::CombinationMultiInputImageToImageMetric()
{
  this->m_NumberOfMetrics = 0;
  this->ComputeGradientOff();

} // end Constructor


/**
 * ********************* PrintSelf ****************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  /** Add debugging information. */
  os << "NumberOfMetrics: "
    << this->m_NumberOfMetrics << std::endl;
  for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    os << "Metric " << i << ":\n";
    os << indent << "MetricPointer: " << this->m_Metrics[ i ].GetPointer() << "\n";
    os << indent << "MetricWeight: " << this->m_MetricWeights[ i ] << "\n";
    os << indent << "MetricValue: " << this->m_MetricValues[ i ] << "\n";
    os << indent << "MetricDerivativesMagnitude: "  << this->m_MetricDerivativesMagnitude[ i ] << "\n";
    os << indent << "MetricComputationTime: " << this->m_MetricComputationTime[ i ] << "\n";
  }

} // end PrintSelf()


/**
 * **************** SetTransform **********************************
 */

template < typename TFixedImage, typename TMovingImage >
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::SetTransform( TransformType * _arg )
{
  this->Superclass::SetTransform( _arg );
  for ( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ )
  {
    if ( i == 0 )
    {
      MultiInputMetricType * testPtr1 = dynamic_cast<MultiInputMetricType *>( this->GetMetric( i ) );
      if ( testPtr1 )
      {
        testPtr1->SetTransform( _arg );
      }
    }
    else 
    {
      ImageMetricType * testPtr2 = dynamic_cast<ImageMetricType *>( this->GetMetric( i ) ); 
      PointSetMetricType * testPtr3 = dynamic_cast<PointSetMetricType *>( this->GetMetric( i ) );
      if ( testPtr2 )
      {
        testPtr2->SetTransform( _arg );
      }
      else if ( testPtr3 )
      {
        testPtr3->SetTransform( _arg );
      }
    }
  }
  
} // end SetTransform()
  
  
/**
 * ******************** SetFixedImageRegion ************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::SetFixedImageRegion( const FixedImageRegionType _arg, unsigned int pos )
{
  if ( pos == 0 ) 
  { 
    for ( unsigned int i = 1; i < this->GetNumberOfMetrics(); i++ ) 
    { 
      ImageMetricType * testPtr1 = dynamic_cast<ImageMetricType *>( this->GetMetric( i ) ); 
	  if ( testPtr1 ) 
      { 
        testPtr1->SetFixedImageRegion( _arg ); 
      } 
    } 
  } 
  MultiInputMetricType * testPtr2 = dynamic_cast<MultiInputMetricType *>( this->GetMetric( 0 ) ); 
  if ( testPtr2 ) 
  { 
    testPtr2->SetFixedImageRegion( _arg, pos ); 
  } 

} // end SetFixedImageRegion()


/**
 * **************** SetFixedImageInterpolator **********************************
 */

template < typename TFixedImage, typename TMovingImage >
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::SetFixedImageInterpolator( FixedImageInterpolatorType *_arg, unsigned int pos )
{
  MultiInputMetricType * testPtr = dynamic_cast<MultiInputMetricType *>( this->GetMetric( 0 ) ); 
  if ( testPtr )
  {
    testPtr->SetFixedImageInterpolator( _arg, pos ); 
  } 
    
} // end SetFixedImageInterpolator()


/**
 * ********************* SetNumberOfMetrics ****************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::SetNumberOfMetrics( unsigned int count )
{
  if ( count != this->m_Metrics.size() )
  {
    this->m_NumberOfMetrics = count;
    this->m_Metrics.resize( count );
    this->m_MetricWeights.resize( count );
    this->m_MetricValues.resize( count );
    this->m_MetricDerivatives.resize( count );
    this->m_MetricDerivativesMagnitude.resize( count );
    this->m_MetricComputationTime.resize( count );
    this->Modified();
  }

} // end SetNumberOfMetrics()


/**
 * ********************* SetMetric ****************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::SetMetric( SingleValuedCostFunctionType * metric, unsigned int pos )
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    this->SetNumberOfMetrics( pos + 1 );
  }

  if ( metric != this->m_Metrics[ pos ] )
  {
    this->m_Metrics[ pos ] = metric;
    this->Modified();
  }

} // end SetMetric()


/**
 * ********************* GetMetric ****************************
 */

template <class TFixedImage, class TMovingImage>
typename CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::SingleValuedCostFunctionType *
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetMetric( unsigned int pos ) const
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    return 0;
  }
  else
  {
    return this->m_Metrics[ pos ];
  }

} // end GetMetric()


/**
 * ********************* SetMetricWeight ****************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::SetMetricWeight( double weight, unsigned int pos )
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    this->SetNumberOfMetrics( pos + 1 );
  }

  if ( weight != this->m_MetricWeights[ pos ] )
  {
    this->m_MetricWeights[ pos ] = weight;
    this->Modified();
  }

} // end SetMetricWeight()


/**
 * ********************* GetMetricWeight ****************************
 */

template <class TFixedImage, class TMovingImage>
double
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetMetricWeight( unsigned int pos ) const
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    return 0.0;
  }
  else
  {
    return this->m_MetricWeights[ pos ];
  }

} // end GetMetricWeight()


/**
 * ********************* GetMetricValue ****************************
 */

template <class TFixedImage, class TMovingImage>
typename CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetMetricValue( unsigned int pos ) const
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    return 0.0;
  }
  else
  {
    return this->m_MetricValues[ pos ];
  }

} // end GetMetricValue()


/**
 * ********************* GetMetricDerivative ****************************
 */

template <class TFixedImage, class TMovingImage>
const typename CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::DerivativeType &
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetMetricDerivative( unsigned int pos ) const
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    return this->m_NullDerivative;
  }
  else
  {
    return this->m_MetricDerivatives[ pos ];
  }

} // end GetMetricDerivative()


/**
 * ********************* GetMetricDerivativeMagnitude ****************************
 */

template <class TFixedImage, class TMovingImage>
double
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetMetricDerivativeMagnitude( unsigned int pos ) const
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    return 0.0;
  }
  else
  {
    return this->m_MetricDerivativesMagnitude[ pos ];
  }

} // end GetMetricDerivativeMagnitude()


/**
 * ********************* GetMetricComputationTime ****************************
 */

template <class TFixedImage, class TMovingImage>
std::size_t
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetMetricComputationTime( unsigned int pos ) const
{
  if ( pos >= this->GetNumberOfMetrics() )
  {
    return 0;
  }
  else
  {
    return this->m_MetricComputationTime[ pos ];
  }

} // end GetMetricComputationTime()


/**
 * **************** GetNumberOfPixelsCounted ************************
 */

template <class TFixedImage, class TMovingImage>
const SizeValueType &
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetNumberOfPixelsCounted( void ) const
{
  unsigned long sum = 0;
  for ( unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i )
  {
    const ImageMetricType * testPtr
      = dynamic_cast<const ImageMetricType *>( this->GetMetric(i) );
    if ( testPtr )
    {
      sum += testPtr->GetNumberOfPixelsCounted();
    }
  }

  this->m_NumberOfPixelsCounted = sum;
  return this->m_NumberOfPixelsCounted;

} // end GetNumberOfPixelsCounted()


/**
 * ********************* Initialize ****************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::Initialize( void ) throw ( ExceptionObject )
{
  /** Check if transform, interpolator have been set. Effectively this
   * method checks if the first sub metric is set up completely.
   * This implicitly means that the first sub metric is an 
   * MultiInputImageToImageMetric, which is a reasonable demand.
   */
  //this->Superclass::Initialize();

  /** Check if at least one (image)metric is provided */
  if ( this->GetNumberOfMetrics() == 0 )
  {
    itkExceptionMacro( << "At least one metric should be set!" );
  }

  /** Call Initialize for all metrics. */
  MultiInputMetricType * testPtr1 = dynamic_cast<MultiInputMetricType *>( this->GetMetric( 0 ) ); 
  if ( testPtr1 )
  {
    testPtr1->Initialize();
  }
  for ( unsigned int i = 1; i < this->GetNumberOfMetrics() ; i++ )
  {
    SingleValuedCostFunctionType * costfunc = this->GetMetric( i );
    if ( !costfunc )
    {
      itkExceptionMacro( << "Metric " << i << " has not been set!" );
    }
    ImageMetricType    * testPtr2 = dynamic_cast<ImageMetricType *>( this->GetMetric( i ) );
    PointSetMetricType * testPtr3 = dynamic_cast<PointSetMetricType *>( this->GetMetric( i ) );
    if ( testPtr2 )
    {
      testPtr2->Initialize();
    }
    else if ( testPtr3 )
    {
      testPtr3->Initialize();
    }
  }

} // end Initialize()


/**
 * ********************* GetValue ****************************
 */

template <class TFixedImage, class TMovingImage>
typename CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const ParametersType & parameters ) const
{
  /** Initialise. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;

  /** Compute, store and combine all metric values. */
  for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    /** Time the computation per metric. */
    itk::TimeProbe timer;
    timer.Start();

    /** Compute ... */
    MeasureType tmpValue = this->m_Metrics[ i ]->GetValue( parameters );
    timer.Stop();

    /** store ... */
    this->m_MetricValues[ i ] = tmpValue;
    this->m_MetricComputationTime[ i ]
      = Math::Round<std::size_t>( timer.GetMean() * 1000.0 );

    /** and combine. */
    measure += this->m_MetricWeights[ i ] * this->m_MetricValues[ i ];
  }

  /** Return a value. */
  return measure;

} // end GetValue()


/**
 * ********************* GetDerivative ****************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative( const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** Initialise. */
  DerivativeType tmpDerivative = DerivativeType( this->GetNumberOfParameters() );
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< MeasureType >::Zero );

  /** Compute, store and combine all metric derivatives. */
  for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    /** Time the computation per metric. */
    itk::TimeProbe timer;
    timer.Start();

    /** Compute ... */
    tmpDerivative.Fill( NumericTraits< MeasureType >::Zero );
    this->m_Metrics[ i ]->GetDerivative( parameters, tmpDerivative );
    timer.Stop();

    /** store ... */
    this->m_MetricDerivatives[ i ] = tmpDerivative;
    this->m_MetricDerivativesMagnitude[ i ] = tmpDerivative.magnitude();
    this->m_MetricComputationTime[ i ]
      = Math::Round<std::size_t>( timer.GetMean() * 1000.0 );

    /** and combine. */
    derivative += this->m_MetricWeights[ i ] * this->m_MetricDerivatives[ i ];
  }

} // end GetDerivative()


/**
 * ********************* GetValueAndDerivative ****************************
 */

template <class TFixedImage, class TMovingImage>
void
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Initialise. */
  MeasureType tmpValue = NumericTraits< MeasureType >::Zero;
  value = NumericTraits< MeasureType >::Zero;

  DerivativeType tmpDerivative = DerivativeType( this->GetNumberOfParameters() );
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< MeasureType >::Zero );

  /** Compute, store and combine all metric values and derivatives. */
  for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    /** Time the computation per metric. */
    itk::TimeProbe timer;
    timer.Start();

    /** Compute ... */
    tmpValue = NumericTraits< MeasureType >::Zero;
    tmpDerivative.Fill( NumericTraits< MeasureType >::Zero );
    this->m_Metrics[ i ]->GetValueAndDerivative( parameters, tmpValue, tmpDerivative );
    timer.Stop();

    /** store ... */
    this->m_MetricValues[ i ] = tmpValue;
    this->m_MetricDerivatives[ i ] = tmpDerivative;
    this->m_MetricDerivativesMagnitude[ i ] = tmpDerivative.magnitude();
    this->m_MetricComputationTime[ i ]
      = Math::Round<std::size_t>( timer.GetMean() * 1000.0 );

    /** and combine. */
    value += this->m_MetricWeights[ i ] * this->m_MetricValues[ i ];
    derivative += this->m_MetricWeights[ i ] * this->m_MetricDerivatives[ i ];
  }

} // end GetValueAndDerivative()


/**
 * ********************* GetMTime ****************************
 */

template <class TFixedImage, class TMovingImage>
unsigned long
CombinationMultiInputImageToImageMetric<TFixedImage,TMovingImage>
::GetMTime( void ) const
{
  unsigned long mtime = this->Superclass::GetMTime();
  unsigned long m;

  // Some of the following should be removed once this 'ivars' are put in the
  // input and output lists

  /** Check the modified time of the sub metrics */
  for ( unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i )
  {
    SingleValuedCostFunctionPointer metric = this->GetMetric( i );
    if ( metric.IsNotNull() )
    {
      m = metric->GetMTime();
      mtime = ( m > mtime ? m : mtime );
    }
  }

  return mtime;

} // end GetMTime()


} // end namespace itk


#undef itkImplementationSetObjectMacro1
#undef itkImplementationSetObjectMacro2

#endif // end #ifndef _itkCombinationMultiInputImageToImageMetric_txx

