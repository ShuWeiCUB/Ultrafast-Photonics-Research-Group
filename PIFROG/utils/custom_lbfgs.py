#%% Adapted from https://github.com/yaroslavvb/stuff/blob/master/eager_lbfgs/eager_lbfgs.py

import tensorflow as tf
import numpy as np
import time
import types


# Time tracking functions
global_time_list = []
global_last_time = 0
def reset_time():
  global global_time_list, global_last_time
  global_time_list = []
  global_last_time = time.perf_counter()
  
def record_time():
  global global_last_time, global_time_list
  new_time = time.perf_counter()
  global_time_list.append(new_time - global_last_time)
  global_last_time = time.perf_counter()
  #print("step: %.2f"%(global_time_list[-1]*1000))

def last_time():
  """Returns last interval records in millis."""
  global global_last_time, global_time_list
  if global_time_list:
    return 1000 * global_time_list[-1]
  else:
    return 0

def dot(a, b):
  """Dot product function since TensorFlow doesn't have one."""
  return tf.reduce_sum(a*b)

def verbose_func(s):
  print(s)

def adaptive_wolfe_line_search(opfunc, x, t_init, d, f0, g0, gtd, opts, barrier, nIter):
    max_ls_iter = opts.get('max_iter', 20)
    alpha = t_init
    tau = 0.5  # backtracking factor
    t_min = 1e-10

    # Convert inputs that might be Python floats to tensors for safe dtype handling
    f0 = tf.convert_to_tensor(f0, dtype=tf.float64)
    gtd = tf.convert_to_tensor(gtd, dtype=tf.float64)
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float64)

    print("alpha " + str(alpha))
    

    # Compute adaptive constants based on current loss
    logf = tf.math.log(f0 + 1e-12)

    #c1 = tf.clip_by_value(1e-2 * tf.exp(logf), 1e-10, 1e-5)
    c1 = 1e-5
    c2 = 0.9
    #c2 = tf.clip_by_value(0.9 * tf.exp(0.5 * (logf - tf.math.log(tf.constant(1e-5, dtype=logf.dtype)))), 0.1, 0.9)
    lsFuncEval = 0
    
    if nIter <=10:
        return f0, g0, x, t_init, lsFuncEval

    

    for i in range(max_ls_iter):
        x_new = x + alpha * d

        barrier.wait()
        result = opfunc(x_new, nIter)
        barrier.wait()

        if result is not None:
            f_new, g_new = result
        else:
            f_new, g_new = None, None

        if result is not None:
            lsFuncEval += 1
            print(f"lsFuncEval {lsFuncEval}")

            gtd_new = tf.reduce_sum(g_new * d)

            print("new loss " + str(f_new))
            print("old loss " + str(f0))
            print("gtd_new " + str(gtd_new))
            print("gtd_old " + str(gtd))
            print("alpha " + str(alpha))
            
            

            # Cast all scalars/tensors to the dtype of f_new to avoid dtype conflicts
            # First ensure they are tensors, then cast
            #f0 = tf.cast(tf.convert_to_tensor(f0,dtype=tf.float64))
            #c1 = tf.cast(tf.convert_to_tensor(c1, dtype=tf.float64))
            #alpha = tf.cast(tf.convert_to_tensor(alpha, dtype=tf.float64))
            #gtd = tf.cast(tf.convert_to_tensor(gtd, dtype=tf.float64))

            # Wolfe conditions
            sufficient_decrease = f_new <= f0 + c1 * alpha * gtd
            curvature_condition = gtd_new >= c2 * gtd

            if sufficient_decrease and curvature_condition:
                break

            # If not satisfied, reduce step size
            alpha *= tau
            if alpha < t_min:
                break

    # Fallback to original step if search fails
    if result is not None:
        if sufficient_decrease and curvature_condition:
            return f_new, g_new, x_new, alpha, lsFuncEval
        else:
            return f0, g0, x, t_init, lsFuncEval
            
    else:
        return None
        
        


final_loss = None
times = []
def lbfgs(opfunc, x, weights, barrier, config, state, do_verbose, log_fn,NeurNet = None):
  """port of lbfgs.lua, using TensorFlow eager mode
  """

  
  if config.maxIter == 0:
    return
  if config.startIter>0:
    iter0 = config.startIter
  else:
    iter0 = 0
    
  global final_loss, times
  
  maxIter = config.maxIter

  maxIter = 1
    
  maxEval = config.maxEval or maxIter*1.25
  tolFun = config.tolFun or 1e-5
  tolX = config.tolX or 1e-19
  nCorrection = config.nCorrection or 100
  lineSearch = config.lineSearch
  lineSearchOpts = config.lineSearchOptions
  learningRate = config.learningRate or 1 
  #learningRate = 2.4

  #lineSearch = adaptive_wolfe_line_search
  #lineSearchOpts = {'max_iter': 20}
  

  print("learningRate "+ str(learningRate))

  isverbose = config.verbose or False

  print("tolX" + str(tolX))

  # verbose function
  if isverbose:
    verbose = verbose_func
  else:
    verbose = lambda x: None

    # evaluate initial f(x) and df/dx
 
    
  barrier.wait()
  #print("passing barrier 1 ")
  result = opfunc(x,0)
  #print("waiting in barrier 2 ")
  barrier.wait()


  if result is not None:
        f,g = result
  else:
    f,g = None, None
    
  nIter = 0+iter0

  f_hist = []
  
  
  

  if result is not None:
      f_hist = [f]
      currentFuncEval = 1
      state.funcEval = state.funcEval + 1

      print("is lbfgs working")

      #p = g.shape[0]
      p = tf.shape(g)[0]  # Works for both TF tensors and NumPy arrays
      #p = np.array(g).shape[0]  # ✅ Convert `ListProxy` to NumPy first


      # check optimality of initial point
      tmp1 = tf.abs(g)
      if tf.reduce_sum(tmp1) <= tolFun:
        verbose("optimality condition below tolFun")
        return x, f_hist

      # optimize for a max of maxIter iterations
      #nIter = 0+iter0
      times = []

      print("maxIter " + str(maxIter))


      sumtime01 = 0
      sumtime12 = 0
      sumtime23 = 0
    
  while nIter < maxIter:
        
    time0 = time.time()
    
    start_time = time.time()
    
    # keep track of nb of iterations
    nIter = nIter + 1
    state.nIter = state.nIter + 1

    
    ############################################################
    ## compute gradient descent direction
    ############################################################
    
    if result is not None:
        
        if state.nIter == 1:
          d = -g
          old_dirs = []
          old_stps = []
          Hdiag = 1
        else:
          # do lbfgs update (update memory)
          y = g - g_old
          s = d*t
          ys = dot(y, s)

          if ys > 1e-10:
            # updating memory
            if len(old_dirs) == nCorrection:
              # shift history by one (limited-memory)
              del old_dirs[0]
              del old_stps[0]

            # store new direction/step
            old_dirs.append(s)
            old_stps.append(y)

            # update scale of initial Hessian approximation
            Hdiag = ys/dot(y, y)

          # compute the approximate (L-BFGS) inverse Hessian 
          # multiplied by the gradient
          k = len(old_dirs)

          if nIter == 2:
              k = 0

          roave = 0
          for i in range(k):
              roave = roave + ro[i]
          if(k != 0): 
              roave = roave/k
          

            
          if(roave < 100):
              roave = 100

          rolimit = 1/roave

          print("roave " + str(roave))
          print("rolimit " + str(1/(rolimit*0.1)))
          # need to be accessed element-by-element, so don't re-type tensor:
          ro = [0]*nCorrection

          

          
          
          for i in range(k):
            #ro[i] = 1/dot(old_stps[i], old_dirs[i])
            #print("ro " + str(i) +" "+ str(ro[i]))
            ro[i] = 1/(max(dot(old_stps[i], old_dirs[i]), rolimit*0.1))
            print("ro " + str(i) +" "+ str(ro[i]))


            

              

            
              
            if(i==-1):
                print("ro 0 "+ str(ro[i]))
            #if(ro[i] == 100):
            #    print(ro[i])
        


          # iteration in L-BFGS loop collapsed to use just one buffer
          # need to be accessed element-by-element, so don't re-type tensor:
          al = [0]*nCorrection

          q = -g
          for i in range(k-1, -1, -1):
            al[i] = dot(old_dirs[i], q) * ro[i]
            q = q - al[i]*old_stps[i]

          # multiply by initial Hessian
          r = q*Hdiag
          for i in range(k):
            be_i = dot(old_stps[i], r) * ro[i]
            r += (al[i]-be_i)*old_dirs[i]

          d = r
          # final direction is in r/d (same object)

        g_old = g
        f_old = f

        #print("f" + str(f))
        #print("g " +

        

    
    ############################################################
    ## compute step length
    ############################################################
    # directional derivative
    
    if result is not None:
        
        time1 = time.time()


        gtd = dot(g, d)

        # check that progress can be made along that direction
        if gtd > -tolX:
          verbose("Can not make progress along direction.")
          break

        # reset initial guess for step size
        if state.nIter == 1:
          tmp1 = tf.abs(g)
          t = min(1, 1/tf.reduce_sum(tmp1))
        else:
          t = learningRate


        # optional line search: user function

        timebeforelinesearch = time.time()
        
        time_weight = time.time()

        lsFuncEval = 0
        #if lineSearch and isinstance(lineSearch) == types.FunctionType:
        if lineSearch and isinstance(lineSearch, types.FunctionType):
           # perform line search, using user function
           print("using line search")
           line_result = lineSearch(opfunc,x,t,d,f,g,gtd,lineSearchOpts, barrier, nIter)

           if line_result is not None:
               f,g,x,t,lsFuncEval = line_result
               print("step size " + str(t))
               if t != 2.4:
                   f_hist.append(f)
          
            
        #if t == 2.4 or nIter <=10:
        else:
          # no line search, simply move with fixed-step

          #print("Dtype of x:", x.dtype)  # e.g., tf.float32 or tf.float64

          #print("Type of x before conversion:", type(x))
          #print("Type of t before conversion:", type(t))
          #print("Type of d before conversion:", type(d))

          #print("Dtype of d:", d.dtype)  # e.g., tf.float32 or tf.float64
        
        
            
          x = tf.convert_to_tensor(x, dtype=tf.float64)
          t = tf.convert_to_tensor(t, dtype=tf.float64)
          d = tf.convert_to_tensor(d, dtype=tf.float64)
        
          
          #print("x tensor" + str(x))
          #print("t tensor" + str(t))
          #print("d tensor" + str(d))



        
 
        
          #print(f"x dtype: {x.dtype}")  # Should be float64
          #print(f"t dtype: {t.dtype}")  # Should be float64
          #print(f"d dtype: {d.dtype}")  # Should be float64
            
          #x = tf.cast(x, dtype=tf.float64)
          #t = tf.cast(t, dtype=tf.float64)
          #d = tf.cast(d, dtype=tf.float64)
        
          x += t*d 
          #print("modifying x")
          #print("x tensor modified" + str(x))

        
          time_numpy1 = time.time()
            
          x_np = x.numpy()
        
          time_numpy2 = time.time()
            
          
          time_numpy3 = time.time()
            
          #for i in range(len(x_np)):
          #      weights[i] = x_np[i]
            
          #print("length of x_np " + str(len(x_np)))
        
          #np.frombuffer(weights.get_obj())[:len(x_np)] = x_np  # ✅ Use `frombuffer()`
          #weights[:len(x_np)] = x_np
          np.frombuffer(weights.get_obj())[:] = x_np

                            
          time_numpy4 = time.time()
        
          timenumpy21 = time_numpy2-time_numpy1
          timenumpy32 = time_numpy3-time_numpy2
          
          #print("time x to numpy " + str(timenumpy21))
          #print("time update weights " + str(timenumpy32))
        
                
        time_weight2 = time.time()
        
        time_weight_time = time_weight2 - time_weight
        #print("update weight time " + str(time_weight_time))
          
          #this is update step of weights
            
          #weights = x 

          #t is the step size/ learning rate, d is the direciton of the gradient so x += t*d gives us the updated "wave". 
          # Then in the main evaluation code we pass x to get the loss value and the value of the gradients g. The gradients g will help 
          # us inform the new direction d of the next step(considering old g and new g, to get hessian matrix etc)
      
    
    ##main evaluation code is here, nIter is epoch
    if nIter != maxIter:
    # re-evaluate function only if not in last iteration
        # the reason we do this: in a stochastic setting,
        # no use to re-evaluate that function here
        
        #print("waiting in barrier 3")
        barrier.wait()
       
        weights_np = np.frombuffer(weights.get_obj(), dtype=np.float64)  # Direct access
        
        x = tf.convert_to_tensor(weights_np)  # Uses shared memory directly
        
        #print("x tensor" + str(x))

        
        #weights_np = np.array(weights)
        #x = tf.convert_to_tensor(weights_np, dtype=tf.float32)
        #redundant if result is not none, if result is none, this is an update step of x
        
        
        #barrier.wait()
        #print("passing barrier 3 counter: ")
        
        result = opfunc(x,nIter)
        #print("waiting in barrier 4")

        barrier.wait() #synchronize cpus
        
        #print("are they waiting in barrier 4")
        
        if result is not None:
            f,g = result
            lsFuncEval = 1
            f_hist.append(f)
            
            
        else:
            f,g = None, None
        

    if result is not None:
        
        timeafterlinesearch = time.time()

        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval

        ############################################################
        ## check conditions
        ############################################################

        time2 = time.time()

        if nIter == maxIter:
          break

        if currentFuncEval >= maxEval:
          # max nb of function evals
          verbose('max nb of function evals')
          break

        tmp1 = tf.abs(g)
        if tf.reduce_sum(tmp1) <=tolFun:
          # check optimality
          verbose('optimality condition below tolFun')
          break

        tmp1 = tf.abs(d*t)
        if tf.reduce_sum(tmp1) <= tolX:
          # step size below tolX
          verbose('step size below tolX')
          break

        if tf.abs(f-f_old) < tolX:
          # function value changing less than tolX
          verbose('function value changing less than tolX'+str(tf.abs(f-f_old)))
          break

        if do_verbose:
          log_fn(nIter, f, NeurNet, is_iter = True)
          #print("Step %3d loss %6.5f msec %6.3f"%(nIter, f.numpy(), last_time()))
          #record_time()
          #times.append(last_time())
          print("loss is " + str(float(f)))
          #print("doing verbose")

        if nIter == maxIter - 1:
          #final_loss = f.numpy()
        
          final_loss = float(f)
          print("final loss " + str(final_loss))
        

        time3 = time.time()

        time01 = time1-time0
        time12 = time2-time1
        time23 = time3-time2


        sumtime01 += time01
        sumtime12 += time12
        sumtime23 += time23

        linesearch_time = timebeforelinesearch - timeafterlinesearch

        #print("time compute gradient descent direction " + str(time01))
        #print("time compute step length " + str(time12))
        #print("time check conditions " + str(time23))


        #print("linesearch_tiume" + str(linesearch_time))
        
        
         
        
        
        




      # save state


  #print("time compute gradient descent direction " + str(sumtime01))
  #print("time compute step length " + str(sumtime12))
  #print("time check conditions " + str(sumtime23))

  if result is not None:
    state.old_dirs = old_dirs
    state.old_stps = old_stps
    state.Hdiag = Hdiag
    state.g_old = g_old
    state.f_old = f_old
    state.t = t
    state.d = d

    return x, f_hist, currentFuncEval
  else:
    pass
    

# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)